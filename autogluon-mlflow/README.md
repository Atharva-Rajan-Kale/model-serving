# AutoGluon MLflow Implementation

Uses native MLflow model serving for AutoGluon inference containers.

## Build and Run

### Prerequisites
You need an AutoGluon model file (`.tar.gz` format) for testing. You can:
- Use an existing model file like `model_1.3.1.tar.gz`
- Train your own model using AutoGluon and export it as a tar.gz file

### Build the Docker Image
```bash
docker build -t autogluon-mlflow:1.3.1-cpu -f Dockerfile.cpu .
```

### Run the Container
```bash
# Create test directory and copy your model file
mkdir -p test_model
cp your_model_file.tar.gz test_model/

# Run the container
docker run -p 5000:5000 -v $(pwd)/test_model:/opt/ml/model autogluon-mlflow:1.3.1-cpu serve
```

**Note**: The `test_model/` directory is gitignored since model files are typically large and shouldn't be in version control.

## API Endpoints

MLflow provides standard model serving endpoints:

- **GET /ping** - Health check endpoint
- **POST /invocations** - Main prediction endpoint
- **GET /version** - Model version information
- **GET /health** - Health status

## Input Formats

### JSON Input (MLflow standard)
```bash
curl -X POST http://localhost:5000/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"feature1": 1.0, "feature2": 2.0}]}'
```

### DataFrame-style Input
```bash
curl -X POST http://localhost:5000/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["feature1", "feature2"], "data": [[1.0, 2.0], [3.0, 4.0]]}}'
```

## Files

- **Dockerfile.cpu** - Docker image with MLflow serving
- **autogluon_model.py** - MLflow PythonModel wrapper for AutoGluon
- **setup_mlflow_model.py** - Script to extract and prepare AutoGluon model for MLflow
- **mlflow-entrypoint.py** - Entry point that sets up and starts MLflow model server
- **README.md** - This documentation

## Features

- **Native MLflow Serving**: Uses `mlflow models serve` command directly
- **AutoGluon Integration**: Custom PythonModel wrapper handles AutoGluon-specific logic
- **Automatic Model Setup**: Extracts tar.gz files and creates proper MLflow model structure
- **Standard MLflow API**: Compatible with MLflow client libraries and tools
- **Model Signatures**: Automatically infers input/output schemas
- **Conda Environment**: Proper dependency management through MLflow

## Architecture

1. **Model Extraction**: `setup_mlflow_model.py` extracts the AutoGluon model from tar.gz
2. **MLflow Wrapper**: `autogluon_model.py` implements `mlflow.pyfunc.PythonModel` interface
3. **Model Registration**: Creates proper MLflow model with artifacts and conda environment
4. **Native Serving**: Uses MLflow's built-in serving infrastructure

## Differences from BentoML/DJLServe

- Uses native MLflow serving instead of custom Flask/Java applications
- Follows MLflow's standard model packaging and serving patterns
- Provides MLflow-native input formats and API endpoints
- Integrates with MLflow ecosystem (tracking, registry, etc.)
- Uses MLflow's built-in health checks and monitoring