# AutoGluon DJLServing Implementation

Replaces TorchServe with DJLServing for AutoGluon inference containers.

## Build and Run

```bash
# Build
docker build -t autogluon-djlserve:1.3.1-cpu -f Dockerfile.cpu .

# Run
mkdir -p test_model
cp ../model_1.3.1.tar.gz test_model/model_1.3.1.tar.gz
docker run -p 8080:8080 -p 8081:8081 -v $(pwd)/test_model:/opt/ml/model autogluon-djlserve:1.3.1-cpu serve
```

## Files

- **Dockerfile.cpu** - Docker image with DJLServing instead of TorchServe
- **serving.properties** - DJLServing configuration
- **djlserving-entrypoint.py** - Entry point that sets up model structure
- **setup_model.sh** - Prepares model structure for DJLServing
- **model.py** - AutoGluon inference script for DJL Python API