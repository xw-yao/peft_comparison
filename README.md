# PEFT Comparison

Comparing a bunch of PEFT methods on T5 and LLaMA.

## Installation instructions

Notice that we require specifically `adapter-transformers` (right before the 2.0)

```bash
pip install -e .
cd adapter-transformers
pip install -e .
cd ..
python -m nltk.downloader punkt
```

If you are running LLaMA fine-tuning, you will probably also need

```bash
huggingface-cli login
```

## Usage example

```
export experiment_name="t5_base_cnn_dailymail_pfeiffer_adapters"                                                                                                          
# python -u -m accelerate.commands.launch --num_processes=2 
python scripts/finetuning_seq2seq.py \
    --output_dir "results/$experiment_name"\
    --model_name_or_path "t5-base" \
    --dataset_name "cnn_dailymail" \
    --adapter_config_string "pfeiffer" \
    --total_batch_size 32 \
    --per_device_train_batch_size 8 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --num_beams 5 \
    --learning_rate 2e-4 \
    --eval_every_steps 1000 \
    --source_prefix "summarize: "
```
