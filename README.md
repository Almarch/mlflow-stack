# mlflow-stack
A minimal stack to play with MLflow locally

### set up

Access the repo:

```sh
git clone https://github.com/almarch/mlflow-stack.git
cd mlflow-stack
```

The following toolbox is needed:

```sh
brew install minikube kubectl k9s helm
sudo apt-get update && sudo apt-get install -y docker.io
```

The k8s instance may then be turned on and off:

```sh
minikube start -p my-mlflow --driver=docker --memory=6144 --cpus=4 --disk-size=50g
minikube stop -p my-mlflow
minikube start -p my-mlflow
```

Generate all secrets:

```sh
echo "POSTGRES_MLFLOW_PASSWORD=$(openssl rand -base64 24 | tr -dc 'A-Za-z0-9' | head -c 16)" > .env
echo "MINIO_ROOT_PASSWORD=$(openssl rand -base64 24 | tr -dc 'A-Za-z0-9' | head -c 16)" >> .env
echo "MINIO_MLFLOW_PASSWORD=$(openssl rand -base64 24 | tr -dc 'A-Za-z0-9' | head -c 16)" >> .env
```

And translate them to k8s:

```sh
kubectl create secret generic mlflow-secrets \
  --from-env-file=.env \
  --dry-run=client -o yaml > k8s/secrets.yaml
kubectl apply -f k8s/secrets.yaml
```

Ingress needs to be activated:

```sh
minikube addons enable ingress -p my-mlflow
```

Build the notebook & mlflow images and load them.

MLflow is light enough and can be built locally then pushed to the cluster:

```sh
docker build -t mlflow:latest -f Dockerfile.mlflow .
minikube image load mlflow:latest -p my-mlflow
```

The notebook is too heavy and must be built from within the cluster:

```sh
eval $(minikube -p my-mlflow docker-env)
docker build -t notebook:latest -f Dockerfile.notebook .
eval $(minikube docker-env -u)
```

Load and deploy all services:

```sh
kubectl apply -f k8s/minio
kubectl apply -f k8s/postgres
kubectl apply -f k8s/mlflow
kubectl apply -f k8s/notebook
```

As this is configured for local deployment, a tunnel is needed:

```sh
minikube tunnel -p my-mlflow
```
