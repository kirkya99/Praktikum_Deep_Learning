docker compose down --remove-orphans

UID=$(id -u)
GID=$(id -g)

export UID
export GID

docker compose -f docker/torch-docker-compose.yaml up --build
