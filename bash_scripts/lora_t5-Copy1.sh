#!/bin/bash

echo "STARTING EXP"

tags= "t5-base" "cb" "classification"

python start_finetuning_lora.py --seed=0 --model_name_or_path="t5-base" --task_name="cb" --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --max_length=128 --learning_rate=4e-4 --lr_scheduler_type="linear" --lr_scheduler_warmup_percent=6e-2 --weight_decay=0 --num_train_epochs=20 --r=8 --lora_alpha=16 --lora_dropout=0 --wandb_project="PEFT" --wandb_name="LoRA_t5-base_cb" --wandb_tags=tags

echo "FINISHED EXP"

echo "STARTING EXP"

tags= "t5-base" "rte" "classification"

python start_finetuning_lora.py --seed=0 --model_name_or_path="t5-base" --task_name="rte" --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --max_length=128 --learning_rate=4e-4 --lr_scheduler_type="linear" --lr_scheduler_warmup_percent=6e-2 --weight_decay=0 --num_train_epochs=20 --r=8 --lora_alpha=16 --lora_dropout=0 --wandb_project="PEFT" --wandb_name="LoRA_t5-base_rte" --wandb_tags=tags

echo "FINISHED EXP"

echo "All experiments completed."