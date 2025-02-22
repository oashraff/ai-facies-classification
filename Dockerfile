# Stage 1: Builder
FROM python:3.9-slim AS builder
WORKDIR /app

# Install OS-level dependencies, including python3-distutils and git-lfs
RUN apt-get update && apt-get install -y python3-distutils git-lfs && rm -rf /var/lib/apt/lists/*
RUN git lfs install

# Copy the entire repository (including .git, if present) into the container
# (Make sure .dockerignore does not exclude .git if you want Git LFS to work)
COPY . .

# If a .git folder exists, pull Git LFS files; otherwise, skip this step.
RUN if [ -d ".git" ]; then git lfs pull && rm -rf .git; else echo "No .git directory, skipping Git LFS pull"; fi

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final Image
FROM python:3.9-slim
WORKDIR /app

# Install runtime OS-level dependencies
RUN apt-get update && apt-get install -y python3-distutils && rm -rf /var/lib/apt/lists/*
# Copy the built application from the builder stage
COPY --from=builder /app /app

EXPOSE 3000

# Command to run your application
CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:3000", "--timeout", "300"]
