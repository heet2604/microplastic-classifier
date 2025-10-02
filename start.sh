#!/bin/bash
# Start Prometheus metrics server in background
python -c "
from prometheus_client import start_http_server, Counter, Histogram, Gauge
start_http_server(8000)
print('✅ Prometheus metrics server started on port 8000')
REQUEST_COUNT = Counter('app_requests_total', 'Total app requests')
DETECTION_COUNT = Counter('detections_total', 'Total microplastic detections')
REQUEST_DURATION = Histogram('app_request_duration_seconds', 'Request latency')
ACTIVE_USERS = Gauge('app_active_users', 'Active users')
APP_START_TIME = Gauge('app_start_time', 'Application start time')
APP_START_TIME.set_to_current_time()
# Keep the metrics server running forever
import time
while True:
    time.sleep(60)
" &

# Start Streamlit app
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
