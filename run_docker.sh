#!/bin/bash

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker command not found" >&2
  exit 1
fi

DOCKERFILE_PATH="docker/Dockerfile.pytorch"
IMAGE_TAG="deim_image"
CONTAINER_TAG="deim_container_${USER}"
FULL_IMAGE="${IMAGE_TAG}:latest"

docker build -f "$DOCKERFILE_PATH" -t "$IMAGE_TAG" .

TZ_OPT="-e TZ=$(cat /etc/timezone)"
PRIV_OPT="--privileged"
NET_OPT="--network host"
DISP_OPT="-e DISPLAY=${DISPLAY}"
HOST_OPT="--hostname deim"
NAME_OPT="--name ${CONTAINER_TAG}"
VOL_OPTS="-v /mnt/d/project_ws:/workspace -v /mnt/d/dataset:/data"
USER_OPT="--user $(id -u):$(id -g)"
GPU_OPT="--gpus all"
CPU_OPT="--ulimit core=-1 --ipc host"

if docker ps -q -f name="${CONTAINER_TAG}" | grep -q .; then
  echo "Attaching to existing container: ${CONTAINER_TAG}"
  docker attach "${CONTAINER_TAG}"
else
  echo "Starting new container: ${CONTAINER_TAG}"
  docker run -it --rm \
    ${CPU_OPT} ${NAME_OPT} ${USER_OPT} ${VOL_OPTS} ${TZ_OPT} \
    ${PRIV_OPT} ${NET_OPT} ${DISP_OPT} ${HOST_OPT} ${GPU_OPT} \
    "${FULL_IMAGE}" /bin/bash
fi
