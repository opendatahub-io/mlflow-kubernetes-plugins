FROM python:3.12-slim

RUN pip install --no-cache-dir "mlflow[extras,db,gateway,genai]" mlflow-kubernetes-plugins

ENTRYPOINT ["mlflow"]
