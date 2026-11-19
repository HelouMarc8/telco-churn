import mlflow

TRACKING_URI = "http://127.0.0.1:5001"
EXPERIMENT_NAME = "telco-churn-tp03"
REGISTERED_MODEL = "telco-churn"
ALIAS = "candidate"


def connect():
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.set_experiment(EXPERIMENT_NAME)


def smoke_run():
    experiment = connect()
    with mlflow.start_run(run_name="smoke-test") as run:
        mlflow.log_param("purpose", "check client-server connection")
    print("Experiment:", experiment.name, "| ID:", experiment.experiment_id)
    print("Run ID:", run.info.run_id)
    print(f"UI: {TRACKING_URI}/#/experiments/{experiment.experiment_id}/runs/{run.info.run_id}")


if __name__ == "__main__":
    smoke_run()
