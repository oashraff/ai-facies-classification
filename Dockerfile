# Stage 1: Builder
FROM python:3.9-slim AS builder
WORKDIR /app

# Install OS-level dependencies, including python3-distutils and git-lfs
RUN apt-get update && apt-get install -y python3-distutils git-lfs && rm -rf /var/lib/apt/lists/*
RUN git lfs install

# Copy the entire repository (including .git) into the container
# Make sure your .dockerignore does not exclude the .git folder
COPY . .

# Pull the actual Git LFS files (the real .pth files) and then remove the .git folder to save space
RUN git lfs pull && rm -rf .git

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final Image
FROM python:3.9-slim
WORKDIR /app

# Install runtime OS-level dependencies
RUN apt-get update && apt-get install -y python3-distutils && rm -rf /var/lib/apt/lists/*
# Copy only the built application from the builder stage
COPY --from=builder /app /app

EXPOSE 3000

# Command to run the FastAPI application using gunicorn with UvicornWorker
CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:3000", "--timeout", "300"]

