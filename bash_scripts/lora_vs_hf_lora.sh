for seed in 11 22 33 42 55
do
    python scripts/finetuning_seq2seq.py \
        --output_dir results/t5-large_super_glue_boolq_lora_lr1e-3_wd0.1_seed${seed} \
        --dataset_name super_glue \
        --dataset_config_name boolq \
        --model_name_or_path t5-large \
        --adapter_config_string lora \
        --per_device_train_batch_size 32 \
        --total_batch_size 32 \
        --max_source_length 512 \
        --max_target_length 8 \
        --num_beams 5 \
        --learning_rate 1e-3 \
        --weight_decay 0.1 \
        --num_train_epochs 3 \
        --min_train_steps 100 \
        --seed ${seed} \
        --tags lora_vs_hf_lora
done
