from mlflow.tracking import MlflowClient

exp = MlflowClient().get_experiment_by_name("mnist-cnn")
run = MlflowClient().search_runs([exp.experiment_id], order_by=["start_time DESC"])[0]
print(run.info.run_id)