# Use NVIDIA CUDA base image (for L4 GPU support)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Install basic tools and Python
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv git curl nano \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Create working directory
WORKDIR /app

# Copy requirements (if separate)
# COPY requirements.txt .

# Install Python packages
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install flask unsloth

# Copy the application files
COPY app.py .

# Expose port
EXPOSE 8080

# Start Flask app using python (not Gunicorn)
CMD ["python", "app.py"]
