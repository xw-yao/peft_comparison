import os
import re
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
stage3_launch_command = "python -m accelerate.commands.launch --config_file accelerate_config_stage3_no_offload.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True, type=str)
    parser.add_argument("--model", type=str, default="t5-large")
    parser.add_argument("--adapter_config_strings", type=str, default=",".join(adapter_config_strings))
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--extra_tags", type=str, default="")

    args = parser.parse_args()

    if args.eval_only:
        logger.info("Will only do eval, no gradient updates. This is useful for the models that are too large for training, but small enough to inference on a single GPU")

    logger.info(f"Starting the script with aruments:")
    logger.info("*" * 40)
    for k, v in vars(args).items():
        logger.info(f"{k:30}: {v}")
    logger.info("*" * 40)

    args.datasets = args.datasets.split(",")
    args.adapter_config_strings = args.adapter_config_strings.split(",")

    errors = []
    started_runs = 0
    experiment_names_ran = []

    for adapter_config_string in args.adapter_config_strings:
        for dataset in args.datasets:
            _time = time.time()
            dataset_name = "super_glue" if dataset in ["rte", "copa", "boolq"] else "cnn_dailymail"
            dataset_config_name = dataset if dataset in ["rte", "copa", "boolq"] else "3.0.0"

            batch_size = hparams[args.model][dataset]["batch_size"]

            max_target_length = 8 if dataset_name != "cnn_dailymail" else 128
            if args.eval_only:
                max_target_length = 1

            n_iters = 51
            max_eval_steps = 21
            if args.eval_only:
                max_eval_steps = 51

            # with some exceptional cases we don't need stage 3 (?)
            launch_command = default_launch_command

            experiment_name = f"{args.model}_{dataset_name}_{dataset_config_name}_{adapter_config_string}"
            experiment_name += args.suffix
            results_path = f"results/{experiment_name}"

            _results_file = results_path + "/all_results.json"
            if os.path.exists(_results_file):
                logger.info(f"{_results_file} already exists, skipping")
                continue

            started_runs += 1
            experiment_names_ran.append(experiment_name)
            # n_iters * batch_size should be a good idea as long as n_iters is small, because batch size is microbatch size
            _tags = f"throughput_estimation"
            if args.eval_only:
                _tags = f"throughput_estimation_eval"
            if args.extra_tags:
                _tags += f",{args.extra_tags}"

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
                            --subsample_data {n_iters * batch_size} \
                            --max_source_length 512 \
                            --max_target_length {max_target_length} \
                            --num_beams 1 \
                            --learning_rate 1e-3 \
                            --weight_decay 0.1 \
                            --min_train_steps {n_iters} \
                            --max_train_steps {n_iters} \
                            --eval_every_steps {n_iters * 2} \
                            --max_eval_steps_durig_validation {max_eval_steps} \
                            --tags "{_tags}" \
            """
            if args.load_in_8bit:
                command += "--load_in_8bit"
            if args.eval_only:
                command += "--eval_throughput_estimation"

            command = re.sub(' +', ' ', command.strip())

            logger.info(f"Running\n{command}")
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            run_name = None
            for line in process.stdout:
                print(line.rstrip())
                if run_name is None and "View run at" in line:
                    run_name = line.split("View run at")[1].strip()
                    logger.info(f"Run name: {run_name}")

            process.wait()

            if process.returncode != 0:
                logger.error(f"Failed to run {experiment_name}, error code {process.returncode}, run name {run_name}")
                logger.error(process.stderr)
                errors.append((experiment_name, run_name, process.stderr))
                continue

            _time = time.time() - _time
            logger.info(f"Finished {experiment_name} in {_time/60:.2f} minutes")

    if started_runs == 0:
        logger.info("No runs were started")
        exit(0)

    wandb.init(project="adapter_throughput")
    if len(errors) > 0:
        table = wandb.Table(columns=["experiment_name", "run_name", "error"])
        for error in errors:
            table.add_data(*error)

        wandb.log({"errors": table})
        logger.error("Finished with errors")
        wandb.alert(
            title=f"Eval Throughput estimation finished. Ran {started_runs} runs with {len(errors)} errors",
            text=f"Ran: {experiment_names_ran}\nErrors: {errors}",
            level=wandb.AlertLevel.WARN,
        )
    else:
        logger.info("Finished successfully")
        wandb.alert(
            title=f"Eval Throughput estimation finished successfully. Ran {started_runs} runs",
            text=f"Ran: {experiment_names_ran}",
            level=wandb.AlertLevel.INFO,
        )
