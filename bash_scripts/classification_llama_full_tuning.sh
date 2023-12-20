set -e

export model="meta-llama/Llama-2-7b-hf"
export dataset_name="super_glue"
export adapter_config_string="full_tuning"
export dataset_config_name="boolq"

export experiment_name="${model}_${dataset_name}_${dataset_config_name}_${adapter_config_string}_debug"

lr=1e-4

python -m accelerate.commands.launch --config_file accelerate_config_stage3_fp16.yaml scripts/finetuning_seq2seq.py \
        --output_dir "results/$experiment_name"\
        --dataset_name $dataset_name \
        --dataset_config_name $dataset_config_name \
        --model_name_or_path $model \
        --adapter_config_string $adapter_config_string \
        --per_device_train_batch_size 1 \
        --total_batch_size 32 \
        --max_source_length 512 \
        --max_target_length 8 \
        --num_beams 5 \
        --learning_rate $lr \
        --num_train_epochs 3 \
        --wandb_project "PEFT_Comparison" \
        --tags "debug" \
