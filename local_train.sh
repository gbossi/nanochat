source .venv/bin/activate 
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- --depth=16 --device_batch_size=4 --target_param_data_ratio=40 --save_every=5000 --run=d16