# Dockerfile

# Use an official NVIDIA CUDA base image. It includes the CUDA toolkit and
# drivers necessary for GPU acceleration on the host.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables to ensure non-interactive installation and
# that Python output is sent straight to the terminal.
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface

# Install Python 3.11, its dev headers, pip, git, and other system dependencies
# --- START OF CORRECTION ---
# Added python3.11-dev to this list. This is the crucial fix.
# It provides the Python.h header file needed for compiling C extensions at runtime.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    git && \
    rm -rf /var/lib/apt/lists/*
# --- END OF CORRECTION ---
   
# Make python3.11 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip, setuptools, and wheel to the latest versions.
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Set up a working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies using pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app.py .

# Expose the port Gunicorn will run on
EXPOSE 8080

# The command to run the application using Gunicorn as a production server.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "600", "app:app"]