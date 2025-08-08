# AutoGluon BentoML Implementation

Replaces DJLServing with BentoML for AutoGluon inference containers.

## Build and Run

### Prerequisites
You need an AutoGluon model file (`.tar.gz` format) for testing. You can:
- Use an existing model file like `model_1.3.1.tar.gz`
- Train your own model using AutoGluon and export it as a tar.gz file

### Build the Docker Image
```bash
docker build -t autogluon-bentoml:1.3.1-cpu -f Dockerfile.cpu .
```

### Run the Container
```bash
# Create test directory and copy your model file
mkdir -p test_model
cp your_model_file.tar.gz test_model/

# Run the container
docker run -p 3000:3000 -v $(pwd)/test_model:/opt/ml/model autogluon-bentoml:1.3.1-cpu serve
```



## Files

- **Dockerfile.cpu** - Docker image with BentoML instead of DJLServing
- **service.py** - Self-contained BentoML service with model extraction and API endpoints
- **bentoml-entrypoint.py** - Entry point that starts BentoML server
