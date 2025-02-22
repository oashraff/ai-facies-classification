# AI Facies Classification End-to-End Application

This repository contains the source code for an AI-powered facies classification web application. The app leverages deep learning models (ResNet34, ResNet50, and InceptionV3 - as backbones) to analyze input images and classify geological facies such as Upper North Sea, Middle North Sea, Lower North Sea, Rijnland/Chalk, Scruff, and Zechstein (using the [Dutch F3 Dataset](https://github.com/yalaudah/facies_classification_benchmark/tree/main)). The application uses Flask for the web interface and FastAPI to provide a RESTful API.

For anyone interested in learning more about the machine learning methodologies, training processes, and model development techniques behind this project, please feel free to reach out for further discussion.

For more technical details or to share insights, contact me at oomaraashrafaabdou@gmail.com.

Visit the app [here](https://ai-facies-classification-production.up.railway.app/)
## Contributors
This project wouldn't have been possible without the hard work and dedication of me and my amazing colleague, Mazen Sakr. I want to take a moment to express my heartfelt gratitude to him.

## Overview

### Web Interface & API
- A user-friendly web interface lets users upload images and view classification results.
- An API endpoint (`/api/predict`) provides predictions along with confidence scores.

### Lazy Model Loading
- Models are loaded asynchronously in the background on the first request, reducing initial startup times.

### Containerized Deployment
- The app is fully containerized using Docker.
- The Dockerfile is designed to work with Git LFS for managing large model weights and supports deployments on platforms such as Koyeb and Railway.

### Health Check Endpoint
- A dedicated `/health` endpoint is provided to verify that the application is running correctly.

## Features

- **Deep Learning Models:** Uses three models (UNet34, UNet50, and UNetInception) to classify facies.
- **Flexible API:** Users can choose the model via a form parameter (default: `resnet50`).
- **Robust Deployment:** Dockerfile configured for multiple deployment environments with Git LFS integration for large model weights.
- **Lazy Loading:** Improves application startup by loading models in a background thread.

## Installation

### Prerequisites

- Python 3.9+
- Git (with [Git LFS](https://git-lfs.github.com/) installed)
- Docker (for containerized deployment)

### Local Setup

#### Clone the Repository

```bash
git clone https://github.com/oashraff/ai-facies-classification.git
cd ai-facies-classification
```

#### Install Dependencies

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Pull Git LFS Files

Ensure Git LFS is installed and run:

```bash
git lfs pull
```

#### Run the Application

```bash
flask run
```

The app will be accessible at [http://127.0.0.1:3000](http://127.0.0.1:3000).

## Docker Deployment

### Building the Docker Image

#### Ensure the `.dockerignore` File

To allow Git LFS to work, remove any exclusions for `.git` and `.pth` files from your `.dockerignore`

#### Build the Image

```bash
docker build -t ai-facies-classification .
```

#### Run the Container Locally

```bash
docker run -it --rm -p 3000:3000 ai-facies-classification
```

## API Documentation

**Endpoint:** `/api/predict`  
**Method:** `POST`

**Parameters:**
- **image** (file): The image to classify.
- **model** (form field, optional): Model to use (`resnet50`, `resnet34`, or `inceptionv3`). Default is `resnet50`.

**Response Example:**

```json
{
  "success": true,
  "prediction": {
    "Upper North Sea": 20.5,
    "Middle North Sea": 30.2,
    "Lower North Sea": 15.0,
    "Rijnland/Chalk": 10.0,
    "Scruff": 12.3,
    "Zechstein": 12.0
  },
  "confidence": 0.87
}
```

