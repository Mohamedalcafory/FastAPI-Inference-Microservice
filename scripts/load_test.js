import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');

// Test configuration
export const options = {
  stages: [
    { duration: '1m', target: 10 },  // Ramp up to 10 users
    { duration: '3m', target: 10 },  // Stay at 10 users
    { duration: '1m', target: 50 },  // Ramp up to 50 users
    { duration: '3m', target: 50 },  // Stay at 50 users
    { duration: '1m', target: 0 },   // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<200'], // 95% of requests must complete below 200ms
    http_req_failed: ['rate<0.1'],    // Error rate must be less than 10%
    errors: ['rate<0.1'],             // Custom error rate
  },
};

// Test data
const testData = {
  single: {
    inputs: [5.1, 3.5, 1.4, 0.2],
    model_version: null
  },
  batch: {
    batches: [
      [5.1, 3.5, 1.4, 0.2],
      [4.9, 3.0, 1.4, 0.2],
      [4.7, 3.2, 1.3, 0.2]
    ],
    model_version: null
  }
};

// Base URL
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export default function () {
  // Test health endpoint
  const healthResponse = http.get(`${BASE_URL}/health`);
  check(healthResponse, {
    'health status is 200': (r) => r.status === 200,
    'health response time < 50ms': (r) => r.timings.duration < 50,
  });

  // Test model info endpoint
  const modelInfoResponse = http.get(`${BASE_URL}/model/info`);
  check(modelInfoResponse, {
    'model info status is 200': (r) => r.status === 200,
    'model info response time < 100ms': (r) => r.timings.duration < 100,
  });

  // Test single prediction
  const singlePredictionResponse = http.post(
    `${BASE_URL}/api/v1/predict`,
    JSON.stringify(testData.single),
    {
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );
  
  const singleCheck = check(singlePredictionResponse, {
    'single prediction status is 200': (r) => r.status === 200,
    'single prediction response time < 200ms': (r) => r.timings.duration < 200,
    'single prediction has prediction field': (r) => {
      try {
        const data = JSON.parse(r.body);
        return 'prediction' in data;
      } catch (e) {
        return false;
      }
    },
  });
  
  if (!singleCheck) {
    errorRate.add(1);
  }

  // Test batch prediction (less frequent)
  if (Math.random() < 0.3) { // 30% of requests
    const batchPredictionResponse = http.post(
      `${BASE_URL}/api/v1/predict/batch`,
      JSON.stringify(testData.batch),
      {
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );
    
    const batchCheck = check(batchPredictionResponse, {
      'batch prediction status is 200': (r) => r.status === 200,
      'batch prediction response time < 500ms': (r) => r.timings.duration < 500,
      'batch prediction has predictions field': (r) => {
        try {
          const data = JSON.parse(r.body);
          return 'predictions' in data;
        } catch (e) {
          return false;
        }
      },
    });
    
    if (!batchCheck) {
      errorRate.add(1);
    }
  }

  // Test async prediction (less frequent)
  if (Math.random() < 0.1) { // 10% of requests
    const asyncPredictionResponse = http.post(
      `${BASE_URL}/api/v1/predict/async`,
      JSON.stringify(testData.single),
      {
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );
    
    const asyncCheck = check(asyncPredictionResponse, {
      'async prediction status is 200': (r) => r.status === 200,
      'async prediction response time < 100ms': (r) => r.timings.duration < 100,
      'async prediction has task_id': (r) => {
        try {
          const data = JSON.parse(r.body);
          return 'task_id' in data;
        } catch (e) {
          return false;
        }
      },
    });
    
    if (!asyncCheck) {
      errorRate.add(1);
    }
  }

  // Test metrics endpoint (occasionally)
  if (Math.random() < 0.05) { // 5% of requests
    const metricsResponse = http.get(`${BASE_URL}/metrics`);
    check(metricsResponse, {
      'metrics status is 200': (r) => r.status === 200,
      'metrics response time < 100ms': (r) => r.timings.duration < 100,
    });
  }

  // Sleep between requests
  sleep(0.1);
}

// Setup function (runs once at the beginning)
export function setup() {
  console.log('Starting load test against:', BASE_URL);
  
  // Verify service is running
  const healthResponse = http.get(`${BASE_URL}/health`);
  if (healthResponse.status !== 200) {
    throw new Error(`Service not ready. Health check returned ${healthResponse.status}`);
  }
  
  console.log('Service is ready for load testing');
}

// Teardown function (runs once at the end)
export function teardown(data) {
  console.log('Load test completed');
}
