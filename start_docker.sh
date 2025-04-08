#!/bin/bash

# docker 명령어가 있는지 확인
if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker command not found" >&2
  exit 1
fi

# 컨테이너 이름 (생성 스크립트와 동일하게 설정)
CONTAINER_TAG="deim_container_${USER}"

# 컨테이너 존재 여부 확인 (모든 상태에서 존재하는지)
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_TAG}$"; then
    # 컨테이너가 존재하면 현재 실행 중인지 확인
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_TAG}$"; then
        echo "Attaching to running container: ${CONTAINER_TAG}"
        docker attach "${CONTAINER_TAG}"
    else
        echo "Starting container: ${CONTAINER_TAG}"
        docker start -ai "${CONTAINER_TAG}"
    fi
else
    echo "Error: Container ${CONTAINER_TAG} does not exist." >&2
    exit 1
fi