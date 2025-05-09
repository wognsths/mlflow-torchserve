import sys, shutil, subprocess, tempfile, pathlib, mlflow
import mlflow.artifacts

run_id = sys.argv[1]
if len(sys.argv) < 2 or not sys.argv[1]:
    print("❌  run_id must be provided :  python make_mar.py <run_id>")
    sys.exit(1)
model_uri = f"runs:/{run_id}/model"

tmpdir = tempfile.mkdtemp()
local_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=tmpdir)
weights = pathlib.Path(local_path) / "data" / "model.pth"

# .mar Packaging
subprocess.run([
    "torch-model-archiver",
    "--model-name", "mnist",
    "--version", "1.0",
    "--serialized-file", str(weights),
    "--handler", "model/handler.py",
    "--export-path", "model_store",
    "--extra-files", "model/model.py"
], check=True)

print("mnist.mar created in model_store/")
shutil.rmtree(tmpdir)