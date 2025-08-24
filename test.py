 
import os
import mlflow

mlflow.set_tracking_uri("https://dagshub.com/2023da04151/yt-comment-analyzer.mlflow")
mlflow.set_experiment("foo_bar_experiment_1")

with mlflow.start_run():
    mlflow.log_param("foo", "bar")