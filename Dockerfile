# Stage 1: Builder
FROM python:3.9-slim AS builder
WORKDIR /app

# Install OS-level dependencies (including distutils and git-lfs)
RUN apt-get update && apt-get install -y python3-distutils git-lfs && rm -rf /var/lib/apt/lists/*
RUN git lfs install

# Copy the entire repository including the .git folder
COPY . .

# Fetch the actual Git LFS files and remove the .git folder to save space
RUN git lfs pull && rm -rf .git

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final Image
FROM python:3.9-slim
WORKDIR /app

# Install runtime OS-level dependencies
RUN apt-get update && apt-get install -y python3-distutils && rm -rf /var/lib/apt/lists/*

# Copy only the built application code from the builder stage
COPY --from=builder /app /app

EXPOSE 3000

# Command to run the FastAPI application
CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:3000"]
