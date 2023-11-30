# batch size table (for 24GB GPU memory):
# t5-base: 8 (maybe 16?)
# t5-large: 4
# t5-3b: 1 (with quantization)
set -e

export model="t5-3b"
export adapter_config_string="houlsby"
export learning_rate=3e-3
export weight_decay=0.1
export seed=1
export experiment_name="${model}_cnn_${adapter_config_string}_lr${learning_rate}_wd${weight_decay}_seed${seed}"
echo "Starting experiment $experiment_name"
python -m accelerate.commands.launch --main_process_port 1235 --num_processes=4 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
    scripts/finetuning_seq2seq.py \
        --output_dir "results/$experiment_name"\
        --dataset_name "cnn_dailymail" \
        --dataset_config_name "3.0.0" \
        --preprocessing_num_workers 12 \
        --model_name_or_path $model \
        --adapter_config_string $adapter_config_string \
        --per_device_train_batch_size 2 \
        --total_batch_size 32 \
        --max_source_length 1024 \
        --max_target_length 128 \
        --num_beams 3 \
        --learning_rate $learning_rate \
        --weight_decay $weight_decay \
        --num_train_epochs 1 \
        --eval_every_steps 2000 \
        --source_prefix "summarize: " \
        --seed $seed \
