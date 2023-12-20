import os
import subprocess
import json
import argparse

import wandb
from loguru import logger

# don't use this script for CNN!
hparams = {
    "t5-large": {
        "rte": {"batch_size": 32},
        "copa": {"batch_size": 32},
        "boolq": {"batch_size": 16},
    },
    "t5-3b": {
        "rte": {"batch_size": 4},
        "copa": {"batch_size": 4},
        "boolq": {"batch_size": 2},
    },
    "t5-11b": {
        "rte": {"batch_size": 1},
        "copa": {"batch_size": 1},
        "boolq": {"batch_size": 1},
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True, type=str)
    parser.add_argument("--model", type=str, default="t5-large")
    parser.add_argument("--adapter_config_string", type=str, default="hf_lora_all")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--launch_command", default=None, type=str)

    args = parser.parse_args()

    model = args.model
    adapter_config_string = args.adapter_config_string
    default_seed = 0
    other_seeds = [1, 42]
    learning_rates = ["1e-3", "1e-4", "5e-5"]
    # datasets = ["rte", "copa", "boolq", "cnn_dailymail"]   # # don't use this script for CNN, it's a bad idea
    datasets = args.datasets.split(",")

    launch_command = "python"
    if args.launch_command is not None:
        launch_command = args.launch_command

    if launch_command == "distributed":
        launch_command = "python -m accelerate.commands.launch --num_processes=8 --main_process_port 1235 --num_machines 1 --mixed_precision bf16 --dynamo_backend no"

    dataset_2_results = {}

    for dataset_name in datasets:
        _dataset_name = "super_glue" if dataset_name in ["rte", "copa", "boolq"] else "cnn_dailymail"
        _dataset_config_name = dataset_name if dataset_name in ["rte", "copa", "boolq"] else "3.0.0"
        _batch_size = hparams[model][dataset_name]["batch_size"]
        _max_target_length = 8 if dataset_name != "cnn_dailymail" else 128
        _train_epochs = 3 if dataset_name != "cnn_dailymail" else 1
        _tag = f"{dataset_name}_{adapter_config_string}_grid_search"

        # small grid over learning rate
        for lr in learning_rates:
            experiment_name = f"{model}_{dataset_name}_{adapter_config_string}_lr{lr}_seed{default_seed}"
            logger.info(f"Running {experiment_name}")
            results_path = f"results/{experiment_name}"

            if os.path.exists(results_path + "/all_results.json"):
                logger.info(f"Skipping {results_path} as it already exists")
                continue

            logger.info(f"Running {experiment_name}")
            command = f"""
                    {launch_command} \
                        scripts/finetuning_seq2seq.py \
                            --output_dir "{results_path}"\
                            --dataset_name "{_dataset_name}" \
                            --dataset_config_name "{_dataset_config_name}" \
                            --model_name_or_path {model} \
                            --adapter_config_string {adapter_config_string} \
                            --per_device_train_batch_size {_batch_size} \
                            --total_batch_size 32 \
                            --max_source_length 512 \
                            --max_target_length {_max_target_length} \
                            --num_beams 5 \
                            --learning_rate {lr} \
                            --weight_decay 0.1 \
                            --num_train_epochs {_train_epochs} \
                            --min_train_steps 100 \
                            --seed {default_seed} \
                            --tags "grid_search,fully_automated,{_tag}" \
            """.strip()
            if args.load_in_8bit:
                command += " --load_in_8bit"

            logger.info(f"Running\n{command}")
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line.strip())
            process.wait()

            if process.returncode != 0:
                raise Exception(f"Error running {experiment_name}, error code {process.returncode}, error message {process.stderr}")

            logger.info(f"Finished {experiment_name}")

        # read results from json and find best lr
        metric_name = "eval/accuracy" if dataset_name in ["rte", "copa", "boolq"] else "eval/rougeL"
        best_lr = learning_rates[0]
        best_metric = float("-inf")

        for lr in learning_rates:
            experiment_name = f"{model}_{dataset_name}_{adapter_config_string}_lr{lr}_seed{default_seed}"
            results_path = f"results/{experiment_name}/all_results.json"

            with open(results_path, "r") as f:
                _results = json.load(f)

            if _results[metric_name] > best_metric:
                best_metric = _results[metric_name]
                best_lr = lr

        # run with best lr
        logger.info(f"Best lr for {dataset_name} is {best_lr} with {metric_name} {best_metric}, running with other seeds")
        run_links = []
        for seed in other_seeds:
            experiment_name = f"{model}_{dataset_name}_{adapter_config_string}_lr{best_lr}_seed{seed}"
            results_path = f"results/{experiment_name}"

            if os.path.exists(results_path + "/all_results.json"):
                logger.info(f"Skipping {results_path} as it already exists")
                continue

            logger.info(f"Running {experiment_name}")
            command = f"""
                    {launch_command} \
                        scripts/finetuning_seq2seq.py \
                            --output_dir "{results_path}"\
                            --dataset_name "{_dataset_name}" \
                            --dataset_config_name "{_dataset_config_name}" \
                            --model_name_or_path {model} \
                            --adapter_config_string {adapter_config_string} \
                            --per_device_train_batch_size {_batch_size} \
                            --total_batch_size {_batch_size} \
                            --max_source_length 512 \
                            --max_target_length {_max_target_length} \
                            --num_beams 5 \
                            --learning_rate {best_lr} \
                            --weight_decay 0.1 \
                            --num_train_epochs {_train_epochs} \
                            --min_train_steps 100 \
                            --seed {seed} \
                            --tags "seeds,fully_automated,{_tag}" \
            """.strip()
            if args.load_in_8bit:
                command += " --load_in_8bit"

            logger.info(f"Running\n{command}")
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line.strip())
                if "View run" in line:
                    run_links.append(line.strip())
            process.wait()

            if process.returncode != 0:
                raise Exception(f"Error running {experiment_name}, error code {process.returncode}, error message {process.stderr}")

            logger.info(f"Finished {experiment_name}")

        # average and std over seeds
        wandb.init(project="automated_peft_comparison")
        logger.info(f"Finished experiments for {dataset_name}, averaging results")
        results_list = []
        all_seeds = [default_seed] + other_seeds
        for seed in all_seeds:
            experiment_name = f"{model}_{dataset_name}_{adapter_config_string}_lr{best_lr}_seed{seed}"
            results_path = f"results/{experiment_name}/all_results.json"

            with open(results_path, "r") as f:
                _results = json.load(f)

            results_list.append(_results[metric_name])

        avg = sum(results_list) / len(results_list)
        std = (sum([(x - avg)**2 for x in results_list]) / len(results_list))**0.5
        logger.info(f"Finished experiments for {model} {dataset_name} {adapter_config_string}, averaging results")
        logger.info(f"Average {metric_name} for {dataset_name} is {avg} with std {std}")
        logger.info(f"Metrics for {dataset_name} are {results_list}")
        logger.info(f"Run links:")
        logger.info('\n'.join(run_links))

        dataset_2_results[dataset_name] = {
            "avg": avg,
            "std": std,
            metric_name: results_list
        }

        try:
            wandb.alert(
                title=f"Finished experiments for {model} {dataset_name} {adapter_config_string}",
                text=(
                    f"Average {metric_name} for {dataset_name} is {avg} with std {std}\n"
                    f"Metrics for {dataset_name} are {results_list}\n"
                    + '\n'.join(run_links)
                ),
                level=wandb.AlertLevel.INFO,
            )
            results_table = wandb.Table(data=[[dataset_name, avg, std]], columns=["dataset", "avg", "std"])
            wandb.log({"results_table": results_table})

            results_table2 = wandb.Table(columns=["dataset", "seed", metric_name])
            for seed, metric in zip(all_seeds, results_list):
                results_table2.add_data(dataset_name, seed, metric)
            wandb.log({"results_full": results_table2})

        except Exception as e:
            logger.info(f"Error sending alert to wandb, error {e}")

        try:
            wandb.finish()
        except Exception as e:
            logger.info(f"Error finishing wandb run, error {e}")

    logger.info(dataset_2_results)
