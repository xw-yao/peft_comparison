# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import json
import math
import os
import random
from typing import List, Optional, Union

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed

import transformers
import datasets
import evaluate
from datasets import load_dataset
from torch.utils.data import DataLoader

from peft import (
    PromptTuningConfig,
    LoraConfig,
    IA3Config,
    get_peft_model,
    PromptTuningInit,
    PromptTuningConfig,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    BitsAndBytesConfig,
)
from transformers.utils.versions import require_version

import wandb
from tqdm.auto import tqdm
from loguru import logger


require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")
datasets.utils.logging.set_verbosity_error()
transformers.utils.logging.set_verbosity_error()

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("premise", "hypothesis"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "boolq": ("passage", "question"),
    "cb": ("premise", "hypothesis"),
    "copa": ("premise", "choice1"),
    "multirc": ("paragraph", "question"),
}

def define_peft_config(args):
    if args.peft_method in ["lora"]:
        peft_config = LoraConfig(
            task_type="SEQ_CLS",
            r=args.r,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            bias=args.bias,
        )

    elif args.peft_method in ["adapters"]:
        peft_config = None

    elif args.peft_method in ["ia_3"]:
        peft_config = IA3Config(
            task_type="SEQ_CLS",
            target_modules=args.target_modules,
            feedforward_modules=args.feedforward_modules,
            init_ia3_weights=args.init_ia3_weights,
        )

    elif args.peft_method in ["prompt_tuning", "p_tuning", "prefix_tuning"]:
        peft_config = PromptTuningConfig(
            task_type="SEQ_CLS",
            prompt_tuning_init=PromptTuningInit.RANDOM,
            num_virtual_tokens=args.num_virtual_tokens,
            tokenizer_name_or_path=args.model_name_or_path,
        )


    return peft_config


def get_model(args, num_labels):

    # check free space
    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
    max_memory = f"{free_in_GB-2}GB"
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}

    # how to load the model
    if args.load_in_4bit:
        args.load_in_8bit = False

    # define model
    if args.use_quantization:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            num_labels=num_labels,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
                llm_int8_threshold=args.llm_int8_threshold,
                bnb_4bit_compute_dtype=args.torch_dtype,
                bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            ),
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            num_labels=num_labels,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
        )

    # freeze the model
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(args.torch_dtype)

    # keeping the model output in float-32 for LM-Head
    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(args.torch_dtype)

    # @TODO: ask Vlad, do we need this for classification?
    if 't5' in args.model_name_or_path:
        model.classification_head = CastOutputToFloat(model.classification_head)
    else:
        model.score = CastOutputToFloat(model.score)

    # define tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is not None:
        logger.info(f"Using present PAD token in the tokenizer: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    else:
        set_pad_to = tokenizer.eos_token
        tokenizer.add_special_tokens({'pad_token': set_pad_to})
        model.config.pad_token_id = model.config.eos_token_id
        logger.info(f"Pointing PAD token to: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")

    # patch the model with PEFT config
    peft_config = define_peft_config(args)
    model = get_peft_model(model, peft_config)
    logger.info(model)

    # Verifying the datatypes.
    logger.info("\nPrecision details:")
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        logger.info(f"{k} {v} {v / total}")

    # verying the size of the model
    par_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    par_fixed = sum(p.numel() for p in model.parameters())
    par_percent = int(100 * par_trainable / par_fixed)
    logger.info(f"Total number of trainable parameters: {par_trainable:,} ({par_percent}%)")

    return model, tokenizer, model.config


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="The name of the glue task to train on.",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="t5-base",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=None,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--total_batch_size",
        type=int,
        default=32,
        help="Total batch size (per_device_batch_size * num_devices * gradient_accumulation)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="(Not recommended, use --total_batch_size instead) Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_scheduler_warmup_percent", type=float, default=0.06, help="Percentage of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--eval_every_steps", type=int, default=None,
        help="Run an evaluation every X steps. By default runs either every 1K steps or 10 times during training (whichever is smaller)."
    )
    parser.add_argument("--output_dir", type=str, default="./results", help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )

    # manually adding arguments for quantization and for LoRA module
    parser.add_argument(
        "--use_quantization",
        type=bool,
        default=False,
        help=("enable 4 or 8bit quantization."),
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help=("Which GPU/s to use for hosting the model"),
    )

    parser.add_argument(
        "--load_in_8bit",
        type=bool,
        default=False,
        help=("enable 8bit quantization."),
    )
    parser.add_argument(
        "--llm_int8_threshold",
        type=float,
        default=6.0,
        help=("value of the outliner threshold. only relevant when load_in_8bit=True"),
    )

    parser.add_argument(
        "--load_in_4bit",
        type=bool,
        default=True,
        help=("enable 4bit quantization."),
    )

    parser.add_argument(
        "--bnb_4bit_quant_type",
        type=str,
        default="fp4",
        help=("set the quantization data type in the `bnb.nn.Linear4Bit` layers. Options are {'fp4','np4'}."),
    )

    parser.add_argument(
        "--bnb_4bit_use_double_quant",
        type=bool,
        default=False,
        help=("enable nested quantization where the quantization constants from the first quantization are quantized again."),
    )

    parser.add_argument(
        "--bnb_4bit_compute_dtype",
        type=bool,
        default="fp16",
        help=(
            "This sets the computational type which might be different than the input time. For example, inputs might be "
            "fp32, but computation can be set to bf16 for speedups. Options are {'fp32','fp16','bf16'}."
        ),
    )

    parser.add_argument(
        "--torch_dtype",
        type=torch.dtype,
        default=torch.bfloat16,
        help=(
            "this sets the dtype of the remaining non quantized layers. `bitsandbytes` library suggests to set the value"
            "to `torch.float16` for 8 bit model and use the same dtype as the compute dtype for 4 bit model "
        ),
    )

    parser.add_argument(
        "--skip_modules",
        type=List[str],
        default=None,
        help=(
            "an explicit list of the modules that we don't quantize. The dtype of these modules will be `torch_dtype`."
        ),
    )

    parser.add_argument(
        "--keep_in_fp32_modules",
        type=List[str],
        default=None,
        help=("an explicit list of the modules that we don't quantize. We keep them in `torch.float32`."),
    )

    # lora arguments
    parser.add_argument(
        "--peft_method",
        type=str,
        default="lora",
        choices=["lora", "prompt_tuning", "p_tuning", "prefix_tuning", "ia_3", "adapters"],
        help=("Lora attention dimension"),
    )

    parser.add_argument(
        "--r",
        type=int,
        default=8,
        help=("Lora attention dimension"),
    )

    parser.add_argument(
        "--target_modules",
        type=str,
        default=None,#["q_proj", "v_proj"],#['k', 'v'],#['q', 'v'],
        help=(
            "List of module names or regex expression of the module names to replace with Lora."
            "For example, 'q,v' or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        ),
    )

    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help=("Lora alpha"),
    )

    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help=("Lora dropout"),
    )

    parser.add_argument(
        "--fan_in_fan_out",
        type=bool,
        default=True,
        help=("Set this to True if the layer to replace stores weight like (fan_in, fan_out)"),
    )

    parser.add_argument(
        "--bias",
        type=str,
        default="none",
        help=("Bias type for Lora. Can be 'none', 'all' or 'lora_only'"),
    )

    parser.add_argument(
        "--modules_to_save",
        type=Optional[List[str]],
        default=None,
        help=(
            "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        ),
    )

    parser.add_argument(
        "--init_lora_weights",
        type=bool,
        default=True,
        help=(
            "Whether to initialize the weights of the Lora layers with their default initialization. Don't change "
            "this setting, except if you know exactly what you're doing."
        ),

    )

    parser.add_argument(
        "--layers_to_transform",
        type=Optional[Union[List[int], int]],
        default=None,
        help=(
            "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes "
            "that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        ),
    )

    parser.add_argument(
        "--layers_pattern",
        type=Optional[str],
        default=None,
        help=(
            "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern "
            "is not in the common layers pattern."
        ),
    )

    # promp-tuning, prefix-tuning and p-tuning arguments
    parser.add_argument(
        "--num_virtual_tokens", 
        type=int,
        default=20,
        help=("how many virtual tokens to add to vocabulary. Embeddings for these tokens will be tuned in the fine-tuning process."),
    )
    parser.add_argument(
        "--prompt_tuning_init", 
        type=str,
        default="text",
        help=("Initialize virtual tokens from text or randomly"),
    )
    parser.add_argument(
        "--prompt_tuning_init_text", 
        type=str,
        default="text",
        help=("If initialization strategy is \"Text\" then, text given to this argument will be used to initialize the virtual tokens"),
    )
    parser.add_argument(
        "--prefix_projection", 
        type=bool,
        default=True,
        help=("Use a two-layer MLP to encode the prefix"),
    )

    # (IA)3 arguments (some arguments are same as LoRA, so not added here)
    parser.add_argument(
        "--feedforward_modules", 
        type=Optional[Union[List[str], str]],
        default=None,
        help=(
            "List of module names or a regex expression of module names which are feedforward" 
            "For example, ['output.dense']"
        ),
    )
    parser.add_argument(
        "--init_ia3_weights", 
        type=bool,
        default=True,
        help=("Whether to initialize the vectors in the (IA)^3 layers."),
    )

    # wandb arguments
    parser.add_argument(
        "--wandb_project", 
        type=str,
        default="PEFT_comparison",
        help=("name to be given to Weight and Biases logging repository"),
    )

    parser.add_argument(
        "--wandb_tags", 
        type=str,
        default=None,
        help=("tags to be given to individual runs in WandB repository, \
              e.g. 'trial, t5-base, classification' "),
    )

    parser.add_argument(
        "--wandb_name", 
        type=str,
        default=None,
        help=("display name for the run"),
    )

    # parse
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.per_device_eval_batch_size is None:
        args.per_device_eval_batch_size = args.per_device_train_batch_size
    
    if args.target_modules:
        if "," in args.target_modules:
            args.target_modules = args.target_modules.split(",")
        if "*" in args.target_modules and "," in args.target_modules:
            raise NotImplementedError("Combining * and , in target_modules is not supported yet.")

    if args.wandb_tags is not None:
        args.wandb_tags = args.wandb_tags.split(",")
        if "classification" not in args.wandb_tags:
            args.wandb_tags.append("classification")

    return args


def evaluate_model(model, *, eval_dataloader, accelerator, is_regression, metric):
    model.eval()
    samples_seen = 0
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
        predictions, references = accelerator.gather((predictions, batch["labels"]))
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(eval_dataloader) - 1:
                predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                references = references[: len(eval_dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]
        metric.add_batch(predictions=predictions, references=references)
    model.train()

    eval_metric = metric.compute()
    return eval_metric


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = Accelerator(log_with="wandb", project_dir=args.output_dir)
    logger.info(f"Started global rank {accelerator.process_index}, device: {torch.cuda.current_device()}")
    args.num_processes = accelerator.num_processes

    if args.num_processes:
        logger.warning("!" * 40)
        logger.warning("Num processes > 1 is not tested. The code assumes that this is the DDP world size and it might not work with model parallel or FSDP")
        logger.warning("!" * 40)

    if args.total_batch_size is not None:
        if args.gradient_accumulation_steps is not None:
            logger.warning("`--total_batch_size` overrides --gradient_accumulation_steps")

        if args.total_batch_size % (accelerator.num_processes * args.per_device_train_batch_size) != 0:
            raise ValueError(f"`--total_batch_size` ({args.total_batch_size}) is not divisible by "
                             f"num_processes * per_device_train_batch_size ({accelerator.num_processes} * {args.per_device_train_batch_size})")

        args.gradient_accumulation_steps = args.total_batch_size // (args.per_device_train_batch_size * accelerator.num_processes)
        logger.info(f"Setting gradient accumulation steps to {args.gradient_accumulation_steps}.")
        # TODO: this won't work with any distributed training, only with DDP
        # If you decide to use fsdp, this needs to be updated!
    else:
        args.total_batch_size = args.gradient_accumulation_steps * args.per_device_train_batch_size * accelerator.num_processes

    if not accelerator.is_main_process:
        logger.remove()

    logger.info("*" * 40)
    logger.info("Accelerator state:")
    logger.info(accelerator.state)
    logger.info("*" * 40)

    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    raw_datasets = load_dataset("super_glue", args.task_name)

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if is_regression:
            num_labels = 1
        else:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    model, tokenizer, config = get_model(args, num_labels)

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts

        if args.task_name == "copa":
            texts = (examples["premise"], examples["choice1"], examples["choice1"], examples["question"])
        elif args.task_name == "multirc":
            texts = (examples["paragraph"], examples["question"], examples["answer"])
        else:
            texts = (examples[sentence1_key],)
            if sentence2_key is not None:
                texts = (examples[sentence1_key], examples[sentence2_key])

        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps
    # NOTE: all of this overcomplicated logic of setting the train steps is needed,
    # because accelerate supports not just data-parallel, but FSDP and so on
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.max_train_steps * args.lr_scheduler_warmup_percent),
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        # NOTE: all of this overcomplicated logic of setting the train steps is needed,
        # because accelerate supports not just data-parallel, but FSDP and so on
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if args.eval_every_steps is None:
        args.eval_every_steps = min(1000, args.max_train_steps // 10)
        args.eval_every_steps = max(100, args.eval_every_steps)
        logger.info(f"Will evaluate every {args.eval_every_steps} steps")

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    experiment_config = vars(args)
    experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
    #accelerator.init_trackers("glue_no_trainer", experiment_config)

    # Get the metric function
    metric = evaluate.load("super_glue", args.task_name)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    update_step = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        logger.info(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            update_step = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            update_step = resume_step // args.gradient_accumulation_step

    # update the progress_bar if load from checkpoint
    progress_bar.update(update_step)

    # start wandb logging
    wandb.init(project=args.wandb_project, config=args)
    global_steps = 0
    active_dataloader = train_dataloader
    if args.resume_from_checkpoint:
        active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        epoch_loss = 0

        for step, batch in enumerate(active_dataloader):
            if update_step >= args.max_train_steps:
                break

            global_steps += 1
            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            epoch_loss += loss.detach().float()

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            is_grad_update = step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1
            if not is_grad_update: continue

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            update_step += 1

            if isinstance(checkpointing_steps, int):
                if update_step % checkpointing_steps == 0:
                    output_dir = f"step_{update_step }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                        # accelerator.save_state(output_dir)

            if update_step >= args.max_train_steps:
                break

            # to wandb
            wandb.log(
                {
                    "Train/loss_per_effective_batch": loss,
                    "Train/learning_rate": optimizer.param_groups[0]["lr"],
                    "Train/epoch": epoch,
                    "Train/par_updates": update_step,
                    "Train/global_steps": global_steps,
                },
                step=update_step,
            )

            if isinstance(checkpointing_steps, int):
                if update_step % checkpointing_steps == 0:
                    output_dir = f"step_{update_step }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if update_step % args.eval_every_steps == 0:
                eval_metric = evaluate_model(
                    model=model,
                    eval_dataloader=eval_dataloader,
                    accelerator=accelerator,
                    is_regression=is_regression,
                    metric=metric,
                )
                logger.info(f"update step {update_step}: {eval_metric}")

                wandb.log(
                    {
                        "Eval/accuracy" if args.task_name is not None else "Eval/glue": eval_metric,
                        "Eval/total_train_loss": epoch_loss / len(train_dataloader),
                        "Eval/epoch": epoch,
                        "Eval/par_updates": update_step,
                        "Eval/global_steps": global_steps,
                    },
                    step=update_step,
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
                # accelerator.save_state(output_dir)

    # final evaluation
    eval_metric = evaluate_model(
        model=model,
        eval_dataloader=eval_dataloader,
        accelerator=accelerator,
        is_regression=is_regression,
        metric=metric,
    )
    logger.info(f"Final evaluation: {eval_metric}")

    wandb.log(
        {
            "Eval/accuracy" if args.task_name is not None else "Eval/glue": eval_metric,
            "Eval/total_train_loss": epoch_loss / len(train_dataloader),
            "Eval/epoch": epoch,
            "Eval/par_updates": update_step,
            "Eval/global_steps": global_steps,
        },
        step=update_step,
    )

    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")
        wandb.log({"Eval/mnli-mm": eval_metric}, step=update_step)

    # save results
    all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
    with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)
    
    # save all arguments
    with open(os.path.join(args.output_dir, "all_inputs.json"), "w") as f:
        args_dict = vars(args)
        for k, v in args_dict.items():
            if not isinstance(v, (float, int, bool, str, list)):
                args_dict[k] = str(v)
        json.dump(args_dict, f, indent=4)

    wandb.save(os.path.abspath(__file__), policy="now")
    accelerator.end_training()

    logger.info("Script finished succesfully")


if __name__ == "__main__":
    main()
