import os
import subprocess
import time
import argparse

import wandb
from loguru import logger

hparams = {
    "t5-large": {
        "rte": {"batch_size": 32},
        "copa": {"batch_size": 32},
        "boolq": {"batch_size": 16},
        "cnn_dailymail": {"batch_size": 4},
    },
    "t5-3b": {
        "rte": {"batch_size": 4},
        "copa": {"batch_size": 4},
        "boolq": {"batch_size": 2},
        "cnn_dailymail": {"batch_size": 1},
    },
    "t5-11b": {
        "rte": {"batch_size": 1},
        "copa": {"batch_size": 1},
        "boolq": {"batch_size": 1},
        "cnn_dailymail": {"batch_size": 1},
    },
}

datasets = [
    ("super_glue", "rte"),
    ("super_glue", "copa"),
    ("super_glue", "boolq"),
    ("cnn_dailymail", "3.0.0"),
]

# "full_tuning" -- different
adapter_config_strings = [
    "houlsby",
    "pfeiffer",
    "scaled_parallel",
    "lora",
    "ln_tuning",
    "hf_lora",
    "hf_lora_all",
    "hf_krona",
    "compacter",
    "compacter++",
    "ia3",
    "mam",
    "prefix_tuning",
    "prefix_tuning_flat",
    "unipelt",
]

default_launch_command = "python"
distributed_launch_command = "python -m accelerate.commands.launch --num_processes=8 --main_process_port 1235 --num_machines 1 --mixed_precision bf16 --dynamo_backend no"
stage3_launch_command = "python -m accelerate.commands.launch --config_file accelerate_config_stage3_no_offload.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True, type=str)
    parser.add_argument("--model", type=str, default="t5-large")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    args = parser.parse_args()

    errors = []

    for adapter_config_string in adapter_config_strings:
        for dataset in args.datasets:
            _time = time.time()
            dataset_name = "super_glue" if dataset in ["rte", "copa", "boolq"] else "cnn_dailymail"
            dataset_config_name = dataset if dataset in ["rte", "copa", "boolq"] else "3.0.0"

            batch_size = hparams[args.model][dataset_name]["batch_size"]

            max_target_length = 8 if dataset_name != "cnn_dailymail" else 128
            n_iters = 50 if dataset_name != "cnn_dailymail" else 300  # enough for throughput to stabilize

            launch_command = default_launch_command if args.model != "t5-11B" else stage3_launch_command

            experiment_name = f"{args.model}_{dataset_name}_{adapter_config_string}"
            results_path = f"results/{experiment_name}"

            if os.path.exists(results_path):
                logger.info(f"{results_path} already exists, skipping")
                continue

            logger.info(f"Running {experiment_name} for {n_iters} iterations")
            command = f"""
                    {launch_command} \
                        scripts/finetuning_seq2seq.py \
                            --output_dir "{results_path}"\
                            --dataset_name "{dataset_name}" \
                            --dataset_config_name "{dataset_config_name}" \
                            --model_name_or_path "{args.model}" \
                            --adapter_config_string "{adapter_config_string}" \
                            --per_device_train_batch_size {batch_size} \
                            --total_batch_size 32 \
                            --subsample_data {n_iters * 32} \
                            --max_source_length 512 \
                            --max_target_length {max_target_length} \
                            --num_beams 3 \
                            --learning_rate 1e-3 \
                            --weight_decay 0.1 \
                            --min_train_steps {n_iters} \
                            --max_train_steps {n_iters} \
                            --tags "throughput_estimation" \
            """.strip()
            if args.load_in_8bit:
                command += " --load_in_8bit"

            logger.info(f"Running\n{command}")
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            run_name = None
            for line in process.stdout:
                print(line.rstrip())
                if run_name is None and "View run at" in line:
                    run_name = line.split("View run at")[1].strip()
                    logger.info(f"Run name: {run_name}")
                    break

            process.wait()

            if process.returncode != 0:
                logger.error(f"Failed to run {experiment_name}")
                logger.error(process.stderr)
                errors.append((experiment_name, run_name, process.stderr))
                continue

            _time = time.time() - _time
            logger.info(f"Finished {experiment_name} in {_time/60:.2f} minutes")

    wandb.init(project="adapter_throughput")
    if len(errors) > 0:
        table = wandb.Table(columns=["experiment_name", "run_name", "error"])
        for error in errors:
            table.add_data(*error)

        wandb.log({"errors": table})
        logger.error("Finished with errors")
        wandb.alert(title=f"Throughput estimation finished with {len(errors)} errors", text=f"{errors}")
    else:
        logger.info("Finished successfully")
        wandb.alert(title="Throughput estimation finished successfully", text="No errors")
