#!/bin/bash

echo "STARTING EXP..."

python start_finetuning_lora.py \
    --seed=0 \
    --model_name_or_path="meta-llama/Llama-2-7b-hf" \
    --task_name="boolq" \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --max_length=128 \
    --learning_rate=1e-4 \
    --lr_scheduler_type="linear" \
    --lr_scheduler_warmup_percent=6e-2 \
    --weight_decay=0 \
    --num_train_epochs=7 \
    --r=8 \
    --lora_alpha=16 \
    --lora_dropout=0 \
    #--device_map="auto" \
    #--use_quantization=true \
    --wandb_project="PEFT_comparison"

echo "FINISHED EXP"


echo "STARTING EXP..."

python start_finetuning_lora.py \
    --seed=0 \
    --model_name_or_path="meta-llama/Llama-2-7b-hf" \
    --task_name="cb" \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --max_length=128 \
    --learning_rate=1e-4 \
    --lr_scheduler_type="linear" \
    --lr_scheduler_warmup_percent=6e-2 \
    --weight_decay=0 \
    --num_train_epochs=10 \
    --r=8 \
    --lora_alpha=16 \
    --lora_dropout=0 \
    #--device_map="auto" \
    #--use_quantization=true \
    --wandb_project="PEFT_comparison"

echo "FINISHED EXP"


echo "STARTING EXP..."


python start_finetuning_lora.py \
    --seed=0 \
    --model_name_or_path="meta-llama/Llama-2-7b-hf" \
    --task_name="copa" \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --max_length=128 \
    --learning_rate=1e-4 \
    --lr_scheduler_type="linear" \
    --lr_scheduler_warmup_percent=6e-2 \
    --weight_decay=0 \
    --num_train_epochs=10 \
    --r=8 \
    --lora_alpha=16 \
    --lora_dropout=0 \
    #--device_map="auto" \
    #--use_quantization=true \
    --wandb_project="PEFT_comparison"

echo "FINISHED EXP"

echo "STARTING EXP..."

python start_finetuning_lora.py \
    --seed=0 \
    --model_name_or_path="meta-llama/Llama-2-7b-hf" \
    --task_name="rte" \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --max_length=128 \
    --learning_rate=1e-4 \
    --lr_scheduler_type="linear" \
    --lr_scheduler_warmup_percent=6e-2 \
    --weight_decay=0 \
    --num_train_epochs=10 \
    --r=8 \
    --lora_alpha=16 \
    --lora_dropout=0 \
    #--device_map="auto" \
    #--use_quantization=true \
    --wandb_project="PEFT_comparison"

echo "FINISHED EXP"






if false; then
    echo "STARTING EXP..."
    
    python start_finetuning_lora.py \
        --seed=0 \
        --model_name_or_path="t5-11b" \
        --task_name="cb" \
        --per_device_train_batch_size=4 \
        --gradient_accumulation_steps=2 \
        --max_length=128 \
        --learning_rate=4e-4 \
        --lr_scheduler_type="linear" \
        --lr_scheduler_warmup_percent=6e-2 \
        --weight_decay=0 \
        --num_train_epochs=10 \
        --r=8 \
        --lora_alpha=16 \
        --lora_dropout=0 \
        --device_map="auto" \
        #--use_quantization=true \
        --wandb_project="PEFT_comparison"
    
    echo "FINISHED EXP"
    
    
    echo "STARTING EXP..."
    
    
    python start_finetuning_lora.py \
        --seed=0 \
        --model_name_or_path="t5-11b" \
        --task_name="copa" \
        --per_device_train_batch_size=4 \
        --gradient_accumulation_steps=2 \
        --max_length=128 \
        --learning_rate=4e-4 \
        --lr_scheduler_type="linear" \
        --lr_scheduler_warmup_percent=6e-2 \
        --weight_decay=0 \
        --num_train_epochs=10 \
        --r=8 \
        --lora_alpha=16 \
        --lora_dropout=0 \
        --device_map="auto" \
        #--use_quantization=true \
        --wandb_project="PEFT_comparison"
    
    echo "FINISHED EXP"
    
    echo "STARTING EXP..."
    
    python start_finetuning_lora.py \
        --seed=0 \
        --model_name_or_path="t5-11b" \
        --task_name="boolq" \
        --per_device_train_batch_size=2 \
        --gradient_accumulation_steps=4 \
        --max_length=128 \
        --learning_rate=1e-4 \
        --lr_scheduler_type="linear" \
        --lr_scheduler_warmup_percent=6e-2 \
        --weight_decay=0 \
        --num_train_epochs=7 \
        --r=8 \
        --lora_alpha=16 \
        --lora_dropout=0 \
        --device_map="auto" \
        #--use_quantization=true \
        --wandb_project="PEFT_comparison"
    
    echo "FINISHED EXP"
    
    
    echo "STARTING EXP..."
    
    python start_finetuning_lora.py \
        --seed=0 \
        --model_name_or_path="t5-11b" \
        --task_name="cb" \
        --per_device_train_batch_size=2 \
        --gradient_accumulation_steps=4 \
        --max_length=128 \
        --learning_rate=1e-4 \
        --lr_scheduler_type="linear" \
        --lr_scheduler_warmup_percent=6e-2 \
        --weight_decay=0 \
        --num_train_epochs=10 \
        --r=8 \
        --lora_alpha=16 \
        --lora_dropout=0 \
        --device_map="auto" \
        #--use_quantization=true \
        --wandb_project="PEFT_comparison"
    
    echo "FINISHED EXP"
    
    
    echo "STARTING EXP..."
    
    
    python start_finetuning_lora.py \
        --seed=0 \
        --model_name_or_path="t5-11b" \
        --task_name="copa" \
        --per_device_train_batch_size=2 \
        --gradient_accumulation_steps=4 \
        --max_length=128 \
        --learning_rate=1e-4 \
        --lr_scheduler_type="linear" \
        --lr_scheduler_warmup_percent=6e-2 \
        --weight_decay=0 \
        --num_train_epochs=10 \
        --r=8 \
        --lora_alpha=16 \
        --lora_dropout=0 \
        --device_map="auto" \
        #--use_quantization=true \
        --wandb_project="PEFT_comparison"
    
    echo "FINISHED EXP"
    
    echo "STARTING EXP..."
    
    python start_finetuning_lora.py \
        --seed=0 \
        --model_name_or_path="t5-11b" \
        --task_name="rte" \
        --per_device_train_batch_size=2 \
        --gradient_accumulation_steps=4 \
        --max_length=128 \
        --learning_rate=1e-4 \
        --lr_scheduler_type="linear" \
        --lr_scheduler_warmup_percent=6e-2 \
        --weight_decay=0 \
        --num_train_epochs=10 \
        --r=8 \
        --lora_alpha=16 \
        --lora_dropout=0 \
        --device_map="auto" \
        #--use_quantization=true \
        --wandb_project="PEFT_comparison"
    
    echo "FINISHED EXP"


    echo "STARTING EXP..."
    
    python start_finetuning_lora.py \
        --seed=0 \
        --model_name_or_path="t5-3b" \
        --task_name="boolq" \
        --per_device_train_batch_size=16 \
        --gradient_accumulation_steps=1 \
        --max_length=128 \
        --learning_rate=1e-4 \
        --lr_scheduler_type="linear" \
        --lr_scheduler_warmup_percent=6e-2 \
        --weight_decay=0 \
        --num_train_epochs=7 \
        --r=8 \
        --lora_alpha=16 \
        --lora_dropout=0 \
        #--device_map="auto" \
        --wandb_project="PEFT_comparison"
    
    echo "FINISHED EXP"
    
    
    echo "STARTING EXP..."
    
    python start_finetuning_lora.py \
        --seed=0 \
        --model_name_or_path="t5-3b" \
        --task_name="cb" \
        --per_device_train_batch_size=16 \
        --gradient_accumulation_steps=1 \
        --max_length=128 \
        --learning_rate=1e-4 \
        --lr_scheduler_type="linear" \
        --lr_scheduler_warmup_percent=6e-2 \
        --weight_decay=0 \
        --num_train_epochs=10 \
        --r=8 \
        --lora_alpha=16 \
        --lora_dropout=0 \
        #--device_map="auto" \
        --wandb_project="PEFT_comparison"
    
    echo "FINISHED EXP"
    
    
    echo "STARTING EXP..."
    
    
    python start_finetuning_lora.py \
        --seed=0 \
        --model_name_or_path="t5-3b" \
        --task_name="copa" \
        --per_device_train_batch_size=16 \
        --gradient_accumulation_steps=1 \
        --max_length=128 \
        --learning_rate=1e-4 \
        --lr_scheduler_type="linear" \
        --lr_scheduler_warmup_percent=6e-2 \
        --weight_decay=0 \
        --num_train_epochs=10 \
        --r=8 \
        --lora_alpha=16 \
        --lora_dropout=0 \
        #--device_map="auto" \
        --wandb_project="PEFT_comparison"
    
    echo "FINISHED EXP"
    
    echo "STARTING EXP..."
    
    python start_finetuning_lora.py \
        --seed=0 \
        --model_name_or_path="t5-3b" \
        --task_name="rte" \
        --per_device_train_batch_size=16 \
        --gradient_accumulation_steps=1 \
        --max_length=128 \
        --learning_rate=1e-4 \
        --lr_scheduler_type="linear" \
        --lr_scheduler_warmup_percent=6e-2 \
        --weight_decay=0 \
        --num_train_epochs=10 \
        --r=8 \
        --lora_alpha=16 \
        --lora_dropout=0 \
        #--device_map="auto" \
        --wandb_project="PEFT_comparison"
    
    echo "FINISHED EXP"


    echo "STARTING EXP..."
    
    python start_finetuning_lora.py \
        --seed=0 \
        --model_name_or_path="t5-large" \
        --task_name="boolq" \
        --per_device_train_batch_size=32 \
        --gradient_accumulation_steps=1 \
        --max_length=128 \
        --learning_rate=4e-4 \
        --lr_scheduler_type="linear" \
        --lr_scheduler_warmup_percent=6e-2 \
        --weight_decay=0 \
        --num_train_epochs=20 \
        --r=8 \
        --lora_alpha=16 \
        --lora_dropout=0 \
        --wandb_project="PEFT_comparison"
    
    echo "FINISHED EXP"
    
    
    echo "STARTING EXP..."
    
    python start_finetuning_lora.py \
        --seed=0 \
        --model_name_or_path="t5-large" \
        --task_name="cb" \
        --per_device_train_batch_size=32 \
        --gradient_accumulation_steps=1 \
        --max_length=128 \
        --learning_rate=4e-4 \
        --lr_scheduler_type="linear" \
        --lr_scheduler_warmup_percent=6e-2 \
        --weight_decay=0 \
        --num_train_epochs=20 \
        --r=8 \
        --lora_alpha=16 \
        --lora_dropout=0 \
        --wandb_project="PEFT_comparison"
    
    echo "FINISHED EXP"
    
    
    echo "STARTING EXP..."
    
    
    python start_finetuning_lora.py \
        --seed=0 \
        --model_name_or_path="t5-large" \
        --task_name="copa" \
        --per_device_train_batch_size=32 \
        --gradient_accumulation_steps=1 \
        --max_length=128 \
        --learning_rate=4e-4 \
        --lr_scheduler_type="linear" \
        --lr_scheduler_warmup_percent=6e-2 \
        --weight_decay=0 \
        --num_train_epochs=20 \
        --r=8 \
        --lora_alpha=16 \
        --lora_dropout=0 \
        --wandb_project="PEFT_comparison"
    
    echo "FINISHED EXP"
    
    echo "STARTING EXP..."
    
    python start_finetuning_lora.py \
        --seed=0 \
        --model_name_or_path="t5-large" \
        --task_name="rte" \
        --per_device_train_batch_size=32 \
        --gradient_accumulation_steps=1 \
        --max_length=128 \
        --learning_rate=4e-4 \
        --lr_scheduler_type="linear" \
        --lr_scheduler_warmup_percent=6e-2 \
        --weight_decay=0 \
        --num_train_epochs=20 \
        --r=8 \
        --lora_alpha=16 \
        --lora_dropout=0 \
        --wandb_project="PEFT_comparison"
    
    echo "FINISHED EXP"


    echo "STARTING EXP..."
    
    python start_finetuning_lora.py \
        --seed=0 \
        --model_name_or_path="t5-base" \
        --task_name="boolq" \
        --per_device_train_batch_size=32 \
        --gradient_accumulation_steps=1 \
        --max_length=128 \
        --learning_rate=5e-4 \
        --lr_scheduler_type="linear" \
        --lr_scheduler_warmup_percent=6e-2 \
        --weight_decay=0 \
        --num_train_epochs=20 \
        --r=8 \
        --lora_alpha=8 \
        --lora_dropout=0 \
        --wandb_project="PEFT_comparison"
    
    echo "FINISHED EXP"
    
    
    echo "STARTING EXP..."
    
    python start_finetuning_lora.py \
        --seed=0 \
        --model_name_or_path="t5-base" \
        --task_name="cb" \
        --per_device_train_batch_size=32 \
        --gradient_accumulation_steps=1 \
        --max_length=128 \
        --learning_rate=5e-4 \
        --lr_scheduler_type="linear" \
        --lr_scheduler_warmup_percent=6e-2 \
        --weight_decay=0 \
        --num_train_epochs=20 \
        --r=8 \
        --lora_alpha=8 \
        --lora_dropout=0 \
        --wandb_project="PEFT_comparison"
    
    echo "FINISHED EXP"
    
    
    echo "STARTING EXP..."
    
    
    python start_finetuning_lora.py \
        --seed=0 \
        --model_name_or_path="t5-base" \
        --task_name="copa" \
        --per_device_train_batch_size=32 \
        --gradient_accumulation_steps=1 \
        --max_length=128 \
        --learning_rate=5e-4 \
        --lr_scheduler_type="linear" \
        --lr_scheduler_warmup_percent=6e-2 \
        --weight_decay=0 \
        --num_train_epochs=20 \
        --r=8 \
        --lora_alpha=8 \
        --lora_dropout=0 \
        --wandb_project="PEFT_comparison"
    
    echo "FINISHED EXP"
    
    echo "STARTING EXP..."
    
    python start_finetuning_lora.py \
        --seed=0 \
        --model_name_or_path="t5-base" \
        --task_name="rte" \
        --per_device_train_batch_size=32 \
        --gradient_accumulation_steps=1 \
        --max_length=128 \
        --learning_rate=5e-4 \
        --lr_scheduler_type="linear" \
        --lr_scheduler_warmup_percent=6e-2 \
        --weight_decay=0 \
        --num_train_epochs=20 \
        --r=8 \
        --lora_alpha=8 \
        --lora_dropout=0 \
        --wandb_project="PEFT_comparison"
    
    echo "FINISHED EXP"
fi


echo "All experiments completed."