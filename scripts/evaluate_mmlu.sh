python3 evaluate_mmlu.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir output/ \
    --ntrain 5 \
    --tau_lambda 0.1 \
    --tau_n 125 \
    --blocksize2 256 \
    --lora_ckpt_dir output/llama-7b-nf4-alpaca-ours/checkpoint-10000
