# batch size table (for 24GB GPU memory):
# t5-base: 8 (maybe 16?)
# t5-large: 4
# t5-3b: 2
set -e

export model="t5-3b"
python -m accelerate.commands.launch --num_processes=2 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
    scripts/finetuning_seq2seq.py \
        --output_dir "results/debug"\
        --dataset_name "cnn_dailymail" \
        --dataset_config_name "3.0.0" \
        --preprocessing_num_workers 12 \
        --model_name_or_path $model \
        --load_in_8bit \
        --adapter_config_string "lora" \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --total_batch_size 32 \
        --max_source_length 1024 \
        --max_target_length 128 \
        --num_beams 5 \
        --learning_rate 2e-4 \
        --num_train_epochs 1 \
        --eval_every_steps 100 \
        --source_prefix "summarize: " \
        --subsample_data 1000 \
        --preprocessing_num_workers 1 \
        --max_eval_steps_durig_validation 2 \
        --tags debug
