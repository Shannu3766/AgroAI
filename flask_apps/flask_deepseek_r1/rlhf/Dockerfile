# Use an official Unsloth image as a base for compatibility
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*
# --------------------------------------------------

# Copy requirements file
COPY requirements.txt .

# --- OPTIONAL BUT RECOMMENDED IMPROVEMENT ---
# Combine all pip installs into a single RUN command to reduce image layers
# This installs from your requirements file AND the other necessary packages in one go.
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install unsloth trl transformers datasets accelerate peft

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Command to run the app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "app:app"]