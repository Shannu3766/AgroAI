import torch.multiprocessing as mp

# Worker settings for GPU
workers = 1
threads = 1
worker_class = 'sync'

# Timeouts
timeout = 900
graceful_timeout = 300
keepalive = 5

# Prevent worker restarts on every request
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Preload app to speed up startup (one worker only)
preload_app = True

def on_starting(server):
    """Ensure 'spawn' start method for CUDA compatibility"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')