# Use an official NVIDIA CUDA base image (with toolkit & driver support)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Make Python output unbuffered, install noninteractive APT, and cache Hugging Face
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH \
    CPATH=/usr/local/cuda/include:$CPATH

# Install Python 3.11 (with dev headers), pip, git, build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.11 \
      python3.11-dev \
      python3-pip \
      python3.11-venv \
      build-essential \
      git && \
    rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip/setuptools/wheel
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8080

# Run with Gunicorn (1 worker, threaded, long timeout for reloads)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", \
     "--workers", "1", \
     "--threads", "8", \
     "--timeout", "600", \
     "app:app"]
