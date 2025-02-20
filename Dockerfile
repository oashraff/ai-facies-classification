# Use the official Python image from Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install OS-level dependencies (including distutils and git-lfs)
RUN apt-get update && apt-get install -y python3-distutils git-lfs && rm -rf /var/lib/apt/lists/*

# Initialize Git LFS
RUN git lfs install

# Copy the entire repository (including the .git directory) into the container
COPY . .

# Fetch the actual Git LFS files (i.e. the real model weights) and then remove .git
RUN git lfs pull && rm -rf .git

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 3000

# Command to run the FastAPI application
CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:3000"]
