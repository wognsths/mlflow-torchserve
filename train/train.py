import mlflow
import mlflow.pytorch
import mlflow.pytorch
import torch, torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from serve.model.model import MNISTModel
from sklearn.metrics import accuracy_score

# Set MLflow experiment
mlflow.set_experiment("mnist-cnn")

# DataLoader
loader = DataLoader(
    datasets.MNIST("./data",
                   train=True,
                   download=True,
                   transform=transforms.ToTensor()),
    batch_size=64,
    shuffle=True
)

model = MNISTModel()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

with mlflow.start_run() as run:
    for epoch in range(10):
        for x, y in loader:
            logits = model(x)
            loss   = criterion(logits, y)
            optim.zero_grad(); loss.backward(); optim.step()
        mlflow.log_metric("train_loss", loss.item(), step=epoch)

    testset = datasets.MNIST("./data", train=False, download=True,
                             transform=transforms.ToTensor())
    testloader = DataLoader(testset, batch_size=1000)
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for x, y in testloader:
            preds.extend(model(x).argmax(1).tolist())
            labels.extend(y.tolist())
    acc = accuracy_score(labels, preds)
    mlflow.log_metric("test_acc", acc)

    mlflow.pytorch.log_model(model, "model")

    print(f"[INFO] Run ID : {run.info.run_id}")