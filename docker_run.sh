#!/bin/bash
cuda=${1:-0}

if [ "$cuda" = "cpu" ]; then
    echo "Running on CPU"
    docker run -it -v "$(pwd)":/aawadalla/code/amorpheus:rw --hostname "$HOSTNAME" --workdir /aawadalla/amorpheus/modular-rl/src/scripts/ amorpheus
else
    echo "Running on GPU: $cuda"
    docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$cuda" -it -v "$(pwd)":/aawadalla/amorpheus:rw --hostname "$HOSTNAME" --workdir /aawadalla/amorpheus/modular-rl/src/scripts/ amorpheus
fi
