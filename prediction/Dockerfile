# Use the official NVIDIA RAPIDS base image
FROM rapidsai/rapidsai:23.04-cuda11.8-runtime-ubuntu22.04-py3.10

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install additional Python libraries
RUN pip install --upgrade pip && pip install flask torch torchvision

# Copy application code
COPY prediction/app /app

# Set the working directory
WORKDIR /app

# Expose port for Flask
EXPOSE 5000

# Set the default command to start a bash shell
CMD ["python", "server.py"]
