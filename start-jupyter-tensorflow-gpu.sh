docker compose down --remove-orphans

UID=$(id -u)
GID=$(id -g)

export UID
export GID

docker compose -f docker/docker-compose.yaml up --build
