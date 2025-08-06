# AutoGluon Inference Model Serving Runtime Migration
A comprehensive implementation repository providing three alternative model serving solutions to replace the deprecated TorchServe runtime for AutoGluon inference containers. 

## Project Context
AutoGluon inference containers currently rely on TorchServe as their primary model serving runtime. However, TorchServe is being deprecated, and PyTorch inference images starting with version 2.7 will no longer support the latest PyTorch versions due to this deprecation. \
This repository provides production-ready implementations of three alternative serving frameworks that maintain full PyTorch compatibility while meeting Amazon's security and licensing requirements.

## Migration Requirements
Any replacement runtime must satisfy these mandatory criteria:

✅ Support for PyTorch framework models and operations \
✅ Ongoing development and security updates \
✅ Meet internal security mechanisms and licensing requirements \
✅ Comparable or superior performance to TorchServe 

## Available Implementations
### 1. DJLServing ⭐ (Recommended)

#### Why DJLServing?

🚀 12% improvement in tabular model inference latency \
🔧 Native TensorRT support and flexible input handling \
🏗️ Built on Amazon's Multi-Model Server (MMS) - same foundation as TorchServe \
🔄 Minimal migration effort with similar REST APIs and deployment patterns \
☁️ Amazon project designed specifically for AWS infrastructure

#### Implementation Includes:

1. Modified Dockerfile with DJLServing installation \
2. Java runtime environment setup \
3. serving.properties configuration (equivalent to config.properties) \
4. Custom model.py handler for DJL serving \
5. djlserving-entrypoint initialization script

### 2. BentoML
A Python-native serving solution with comprehensive ML framework support.

#### Implementation Includes:

1. Service-oriented architecture with @svc.api decorators \
2. Custom runners for AutoGluon-specific models \
3. Input/output adapters for API compatibility \
4. YAML-based configuration system \
5. Health check and metrics collection endpoints

### 3. MLflow
Enterprise-grade model serving with comprehensive MLOps integration.

#### Implementation Includes:

1. MLflow model format conversion utilities \
2. Standardized REST /invocations endpoint \
3. Model signature and schema management \
4. Input/output transformation layers \
5. Model registry and versioning integration 
 
## Quick Start
```script
# Clone the repository
git clone https://github.com/Atharva-Rajan-Kale/model-serving.git
cd model-serving
```

## Additional Resources
For detailed implementation-specific information, configuration options, and advanced usage examples, please refer to the README files located in each directory:

DJLServing: See djlserving/README.md for detailed setup, configuration, and troubleshooting \
BentoML: See bentoml/README.md for service implementation details and deployment options \
MLflow: See mlflow/README.md for model format conversion and MLOps integration guides