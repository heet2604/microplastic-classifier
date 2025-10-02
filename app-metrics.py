# Replace your Prometheus metrics section with:
try:
    from prometheus_client import start_http_server, Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
    
    # Use UNIQUE metric names to avoid conflicts
    MICROPLASTIC_REQUEST_COUNT = Counter('microplastic_requests_total', 'Total app requests')
    MICROPLASTIC_DETECTION_COUNT = Counter('microplastic_detections_total', 'Total microplastic detections')
    MICROPLASTIC_REQUEST_DURATION = Histogram('microplastic_request_duration_seconds', 'Request latency')
    MICROPLASTIC_ACTIVE_USERS = Gauge('microplastic_active_users', 'Active users')
    
    # Start metrics server on port 8000
    start_http_server(8000)
    print("✅ Prometheus metrics server started on port 8000")
    
except Exception as e:
    print(f"⚠️ Prometheus disabled: {e}")
    PROMETHEUS_AVAILABLE = False
    # Create dummy metrics as fallback
    class DummyMetric:
        def inc(self, n=1): pass
        def set(self, value): pass
        def observe(self, value): pass
    MICROPLASTIC_REQUEST_COUNT = MICROPLASTIC_DETECTION_COUNT = MICROPLASTIC_REQUEST_DURATION = MICROPLASTIC_ACTIVE_USERS = DummyMetric()

# Update your tracking functions:
def track_detection(count=1):
    if PROMETHEUS_AVAILABLE:
        MICROPLASTIC_DETECTION_COUNT.inc(count)

def track_request():
    if PROMETHEUS_AVAILABLE:
        MICROPLASTIC_REQUEST_COUNT.inc()