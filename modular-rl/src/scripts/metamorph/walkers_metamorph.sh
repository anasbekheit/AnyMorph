cd ../..
for seed in 1 2 3;
do
	sleep 5
        python3 main.py \
          --custom_xml environments/walkers \
          --actor_type metamorph \
          --critic_type metamorph \
          --seed $seed \
          --grad_clipping_value 0.1 \
          --lr 0.0001 \
          --transformer_norm 1 \
          --attention_embedding_size 128 \
          --attention_heads 2 \
          --attention_hidden_size 1024 \
          --attention_layers 5 \
          --dropout_rate 0.1 \
          --label walkers_metamorph&
done
cd scripts