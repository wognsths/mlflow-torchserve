# mlflow-torchserve
Tracking machine learning flow with mlflow, and serving with torchserve

## MLFLOW
![alt text](assets/mlflow-experiments.png.png)
### Mlflow ui 접속
```bash
mlflow ui
```
### 모델 학습
```bash
python train.py
```
###




## TORCHSERVE
### .mar file 만들기 (제공)
```bash
torch-model-archiver   --model-name mnist_model   --version 1.0   --serialized-file serve/mnist_model_state.pt   --handler serve/handler.py   --extra-files "serve/model.py,serve/__init__.py"   --export-path model_store   --force
```

### torchserve 실행
```bash
torchserve --start   --model-store model_store   --models mnist_model=mnist_model.mar   --disable-token-auth
```

### torchserve 중지
```bash
torchserve --stop
```