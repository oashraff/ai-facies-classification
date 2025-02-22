# Stage 1: Builder
FROM python:3.9-slim AS builder
WORKDIR /app

# Install OS-level dependencies and Git LFS
RUN apt-get update && apt-get install -y python3-distutils git-lfs && rm -rf /var/lib/apt/lists/*
RUN git lfs install

# Clone the repository so that .git is present
RUN git clone https://github.com/oashraff/ai-facies-classification.git .

# Pull the Git LFS files
RUN git lfs pull

# Optionally remove the .git folder if it's not needed later
RUN rm -rf .git

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final image
FROM python:3.9-slim
WORKDIR /app

# Install runtime OS-level dependencies
RUN apt-get update && apt-get install -y python3-distutils && rm -rf /var/lib/apt/lists/*

# Copy installed packages and application code from builder
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

EXPOSE 3000

CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:3000", "--timeout", "300"]
