#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

experiment_name="t5_3b_cb_lora"
if [[ -f results/$experiment_name/all_results.json ]]; then
    echo "Experiment $experiment_name is already completed. Continuing to next experiment."
else
    echo "STARTING EXP $experiment_name..."
    python scripts/finetuning_classification.py \
        --output_dir "results/$experiment_name"\
        --seed 0 \
        --model_name_or_path "t5-3b" \
        --task_name "cb" \
        --peft_method "lora" \
        --r 8 \
        --lora_alpha 16 \
        --lora_dropout 0 \
        --target_modules "q, v" \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --total_batch_size 8 \
        --use_quantization true \
        --load_in_4bit true \
        --max_length 128 \
        --learning_rate 0.0001 \
        --lr_scheduler_type "linear" \
        --lr_scheduler_warmup_percent 0.06 \
        --weight_decay 0 \
        --num_train_epochs 20 \
        --wandb_project "PEFT_comparison" \

    echo "FINISHED EXP $experiment_name!!!"
fi


experiment_name="t5_3b_copa_lora"
if [[ -f results/$experiment_name/all_results.json ]]; then
    echo "Experiment $experiment_name is already completed. Continuing to next experiment."
else
    echo "STARTING EXP $experiment_name..."
    python scripts/finetuning_classification.py \
        --output_dir "results/$experiment_name"\
        --seed 0 \
        --model_name_or_path "t5-3b" \
        --task_name "copa" \
        --peft_method "lora" \
        --r 8 \
        --lora_alpha 16 \
        --lora_dropout 0 \
        --target_modules "q, v" \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --total_batch_size 8 \
        --use_quantization true \
        --load_in_4bit true \
        --max_length 128 \
        --learning_rate 0.0001 \
        --lr_scheduler_type "linear" \
        --lr_scheduler_warmup_percent 0.06 \
        --weight_decay 0 \
        --num_train_epochs 20 \
        --wandb_project "PEFT_comparison" \

    echo "FINISHED EXP $experiment_name!!!"
fi
