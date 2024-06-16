#!/bin/bash

NUM_GPUS=4

main() {
    cd ../../.. || exit 1

    for bc_value in 0 1; do
        for ENV_NAME in hoppers; do
            for resample_value in 1 4 8; do
                for seed in $(seq 1 $NUM_GPUS); do
                    sleep 5

                    GPU_ID=$((seed % 2))

                    CUDA_VISIBLE_DEVICES=$GPU_ID python3 main.py \
                      --custom_xml environments/${ENV_NAME} \
                      --env_name ${ENV_NAME} \
                      --actor_type metamorph \
                      --critic_type metamorph \
                      --seed $seed \
                      --grad_clipping_value 0.1 \
                      --priority_buffer 0 \
                      --lr 0.0001 \
                      --transformer_norm 1 \
                      --attention_embedding_size 128 \
                      --attention_heads 2 \
                      --attention_hidden_size 1024 \
                      --attention_layers 5 \
                      --dropout_rate 0.1 \
                      --bc $bc_value \
                      --alpha 2.5 \
                      --resample $resample_value \
                      --label ${ENV_NAME}_metamorph_bc${bc_value}_rs${resample_value} &
                done
            done
        done
    done
    cd scripts || exit 1
}

main "$@"