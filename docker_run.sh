#!/bin/bash
cuda=${1:-0}

if [ "$cuda" = "cpu" ]; then
    echo "Running on CPU"
    docker run -it -v "$(pwd)":/$USER/code/amorpheus:rw --hostname "$HOSTNAME" --workdir /$USER/amorpheus/modular-rl/src/scripts/ amorpheus
else
    echo "Running on GPU: $cuda"
    docker run --runtime=nvidia --gpus all -it -v "$(pwd)":/$USER/amorpheus:rw --hostname "$HOSTNAME" --workdir /$USER/amorpheus/modular-rl/src/scripts/ amorpheus
fi
