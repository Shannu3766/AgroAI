import multiprocessing

# Worker settings
workers = 1  # Single worker for GPU applications
threads = 1
worker_class = 'sync'
worker_connections = 1000

# Prevent timeout issues
timeout = 900
graceful_timeout = 300
keepalive = 5

# Reduce memory leaks
max_requests = 1
max_requests_jitter = 0

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Process naming
proc_name = 'deepseek-flask'

# Startup configurations
preload_app = True
daemon = False

def on_starting(server):
    """Called just before the master process is initialized."""
    import torch.multiprocessing as mp
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')