# batch size table (for 24GB GPU memory and classification tasks):
# t5-base: 32
# t5-large: 4
# t5-3b: 1 (with quantization)
set -e

export model="t5-large"
export dataset_name="super_glue"
for adapter_config_string in \
    "ln_tuning" "compacter" "compacter++" "ia3" \
    "pfeiffer" "houlsby" "scaled_parallel"
    # "lora"
    # "unipelt" "prefix_tuning" "prefix_tuning_flat" "mam"
do

for dataset_config_name in \
    "cb" "copa"
do

    export experiment_name="${model}_${dataset_name}_${dataset_config_name}_${adapter_config_string}"

    if [[ -f results/$experiment_name/all_results.json ]]; then
        echo "Experiment $experiment_name is already completed. Continuing to next experiment."
        continue
    fi

    lr=2e-4
    if adapter_config_string == "ai3"; then
        lr=1e-3
    fi

    echo "Starting experiment $experiment_name"
    # python -m accelerate.commands.launch --num_processes=2 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
    python scripts/finetuning_seq2seq.py \
            --output_dir "results/$experiment_name"\
            --dataset_name $dataset_name \
            --dataset_config_name $dataset_config_name \
            --model_name_or_path $model \
            --load_in_4bit \
            --adapter_config_string $adapter_config_string \
            --per_device_train_batch_size 4 \
            --total_batch_size 32 \
            --max_source_length 512 \
            --max_target_length 8 \
            --num_beams 5 \
            --learning_rate $lr \
            --num_train_epochs 3 \
            --min_train_steps 100 \
            --wandb_project "PEFT_Comparison" \
            --tags "min_train_steps_100" \

done
done