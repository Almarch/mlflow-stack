# mlflow-stack
A minimal stack to play with MLflow locally

### set up

```sh
echo "POSTGRES_MLFLOW_PASSWORD=$(openssl rand -base64 24 | tr -dc 'A-Za-z0-9' | head -c 16)" > .env
echo "MINIO_ROOT_PASSWORD=$(openssl rand -base64 24 | tr -dc 'A-Za-z0-9' | head -c 16)" >> .env
echo "MINIO_MLFLOW_PASSWORD=$(openssl rand -base64 24 | tr -dc 'A-Za-z0-9' | head -c 16)" >> .env
```

then:

```sh
docker compose up -d
```


