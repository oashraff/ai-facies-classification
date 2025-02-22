# Stage 1: Builder
FROM python:3.9-slim AS builder
WORKDIR /app

# Install OS-level dependencies and Git LFS
RUN apt-get update && apt-get install -y python3-distutils git-lfs && rm -rf /var/lib/apt/lists/*
RUN git lfs install

# Copy source code
COPY . .

# If a .git folder exists, pull Git LFS files and remove .git
RUN if [ -d ".git" ]; then git lfs pull && rm -rf .git; else echo "No .git directory, skipping Git LFS pull"; fi

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final image
FROM python:3.9-slim
WORKDIR /app

# Install runtime OS-level dependencies
RUN apt-get update && apt-get install -y python3-distutils && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder (assuming they went into /usr/local)
COPY --from=builder /usr/local /usr/local
# Copy your application code
COPY --from=builder /app /app

EXPOSE 3000

CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:3000", "--timeout", "300"]
