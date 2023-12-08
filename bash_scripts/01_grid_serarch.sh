set -e

export model="t5-large"
export dataset_name="cnn_dailymail"

learning_rates=(1e-3 1e-4 5e-5)
weight_decays=(0 0.1)
seed_vals=(0 1 42)

for adapter_config_string in "unipelt" "prefix_tuning" "prefix_tuning_flat"; do
    for seed in "${seed_vals[@]}"; do
        for dataset_config_name in "3.0.0"; do
            for lr in "${learning_rates[@]}"; do
                for weight_decay in "${weight_decays[@]}"; do

                    export experiment_name="${model}_${dataset_name}_${dataset_config_name}_${adapter_config_string}_lr${lr}_wd${weight_decay}_seed${seed}"

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
                            --total_batch_size 64 \
                            --max_source_length 512 \
                            --max_target_length 128 \
                            --num_beams 3 \
                            --learning_rate $lr \
                            --weight_decay $weight_decay \
                            --num_train_epochs 1 \
                            --min_train_steps 100 \
                            --seed $seed \
                            --tags "grid_search, t5-large, cnn" \

                done
            done
        done
    done
done