export HF_ENDPOINT="https://hf-mirror.com"
# export HF_HOME="/path/to/hf_cache"
# export HUGGINGFACE_HUB_CACHE="/path/to/hf_cache"
# export HUGGING_FACE_TOKEN="your_token_here"
python ircot_evaluation.py \
        --model_path ./checkpoints/global_step199 \
        --dataset_name musique \
        --split train \
        --tokenizer_path Qwen/Qwen3-0.6B \
        --index_path ./results/musique_document_index_new.pt \
        --output_dir ./results