set -e

export model="t5-3b"
export dataset_name="super_glue"
export adapter_config_string="houlsby"

lr=1e-3
weight_decay=0.1
dataset_config_name="rte"

for seed in 0 1 42; do
    experiment_name="${model}_${dataset_name}_${dataset_config_name}_${adapter_config_string}_lr${lr}_wd${weight_decay}_seed${seed}"

    if [[ -f results/$experiment_name/all_results.json ]]; then
        echo "Experiment $experiment_name is already completed. Continuing to next experiment."
        continue
    fi

    echo "Starting experiment $experiment_name with LR: $lr, Weight Decay: $weight_decay"

    python scripts/finetuning_seq2seq.py \
            --output_dir "results/$experiment_name"\
            --dataset_name $dataset_name \
            --dataset_config_name $dataset_config_name \
            --model_name_or_path $model \
            --adapter_config_string $adapter_config_string \
            --per_device_train_batch_size 16 \
            --total_batch_size 32 \
            --max_source_length 512 \
            --max_target_length 8 \
            --num_beams 5 \
            --learning_rate $lr \
            --weight_decay $weight_decay \
            --num_train_epochs 3 \
            --min_train_steps 100 \
            --tags "varying_seed" \
            --seed $seed

done
