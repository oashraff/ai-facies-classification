# Stage 1: Build Stage
FROM python:3.9-slim AS builder
WORKDIR /app

# Install build dependencies (including Git LFS)
RUN apt-get update && apt-get install -y python3-distutils git-lfs && rm -rf /var/lib/apt/lists/*

# Initialize Git LFS
RUN git lfs install

# Copy only files necessary for building (your .dockerignore will exclude unnecessary files)
COPY requirements.txt .
COPY . .

# Fetch the actual model weights from Git LFS and remove the .git folder to avoid bloating the final image
RUN git lfs pull && rm -rf .git

# Install Python dependencies (this layer may include some build artifacts; they won’t be copied later if not needed)
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final Image
FROM python:3.9-slim
WORKDIR /app

# Install runtime OS dependencies
RUN apt-get update && apt-get install -y python3-distutils && rm -rf /var/lib/apt/lists/*

# Copy only the built application code and files from the builder stage
COPY --from=builder /app /app

EXPOSE 3000

# Command to run your application
CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:3000"]
