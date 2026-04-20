export HF_ENDPOINT="https://hf-mirror.com"
# export HF_HOME="/path/to/hf_cache"
# export HUGGINGFACE_HUB_CACHE="/path/to/hf_cache"
# export HUGGING_FACE_TOKEN="your_token_here"
# python ./download_model.py
deepspeed --num_gpus 1 multihop_contrastive_train.py --train_batch_size 256 --model_name Qwen/Qwen3-0.6B