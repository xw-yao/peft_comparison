import re
from functools import partial

import nltk
import torch
from datasets import DatasetDict, Dataset

from peft_comparison.mappings import (
    summarization_name_mapping,
    clf_label_names_mapping,
    task_to_keys,
    clf_task_description_mapping,
)


# Preprocessing functions
def dataset_to_text2text(dataset, task_type, dataset_name, decoder_only=False):
    """Takes a dataset dict and returns a dataset with fields source_text target_text and a postprocessing function

    If task_type is summarization, just renames the columns
    If task_type is classification, applies `preprocess_glue_one_example` to convert
    classification to text generation

    Args:
        dataset: hf dataset or hf dataset dict
        task_type: str, "summarization" or "classification"
        dataset_name: name of a summarization dataset
            from peft_comparison.mappings.summarization_name_mapping
            (e.g., "cnn_dailymail")
            OR
            name of a glue/superglue task name from clf_label_names_mapping (e.g., "cola")
            NOT "superglue" or "glue"
    """
    if task_type == "summarization":
        if not decoder_only:
            source_text_column, target_text_column = summarization_name_mapping[dataset_name]
            dataset = dataset.rename_column(source_text_column, "source_text")
            dataset = dataset.rename_column(target_text_column, "target_text")
        else:
            NotImplementedError("Summarization tasks: data preprocessing for decoder-only models is not implemented yet.")
        return dataset, postprocess_summarization

    if task_type != "classification":
        raise ValueError(f"Unknown task type: {task_type}")

    # task_type == classification from now on
    if dataset_name not in clf_label_names_mapping:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Note that for classification, dataset_name must be a glue/superglue task name (e.g., 'cola')")
    dataset = dataset.map(
        preprocess_glue_one_example,
        batched=False,
        fn_kwargs={"task_name": dataset_name, "decoder_only": decoder_only},
    )

    if isinstance(dataset, dict):
        for subset in dataset.values():
            assert "source_text" in subset.column_names, dataset
            assert "target_text" in subset.column_names, dataset
    else:
        assert "source_text" in dataset.column_names
        assert "target_text" in dataset.column_names

    postprocess_fn = partial(postprocess_classification, dataset_config_name=dataset_name)

    return dataset, postprocess_fn



def preprocess_glue_one_example(x, task_name, id_key="idx", decoder_only=False):
    """

    CODE SOURCE: https://github.com/google-research/text-to-text-transfer-transformer/blob/24d9d3b89b129e586bbfe35cffbc5926d88adc5e/t5/data/preprocessors.py#L734C1-L812C12

    Convert a dataset from glue to text2text examples.

    This function uses the feature names from the dataset to unpack examples into
    a format amenable for a text2text problem. For example, consider the Quora
    Question Pairs (QQP) benchmark, which would suggest
    benchmark_name="qqp"
    label_names=["not_duplicate", "duplicate"]
    For QQP, a typical example might look like
    {
        "question1": "Why do I easily get bored of my friends?",
        "question2": "Why do I get bored of friends so quickly?",
        "label": 1,
        "idx": 10,
    }

    This example would be transformed to
    {
        "inputs": (
            "qqp question1: Why do I easily get bored of my friends? question2: "
            "Why do I get bored of my friends so quickly?"
        ),
        "targets": "duplicate",
        "idx": 10,
    }

    Args:
    x: an example to process.
    benchmark_name: the name of the GLUE benchmark for this dataset.
    label_names: a list of label names corresponding to class index.
    feature_names: an optional ordered list of feature names. If provided,
        features will be ordered in this way in the output. If not provided, all
        features (except "idx" and "label") will be used, sorted by name.
    id_key: str, key for id in the dataset. If not provided, "idx" will be used.
        if None, no id will be added to the dataset.

    Returns:
    A preprocessed example.
    """
    feature_names = task_to_keys[task_name]
    label_names = clf_label_names_mapping[task_name]
    input_text = f"{task_name} "
    for feature_name in feature_names:
        if feature_name is None: continue
        input_text += f"{feature_name}: {x[feature_name]} "

    # label name
    if x["label"] == -1:
        label_name = "<unk>"
    else:
        label_name = label_names[x["label"]]

    ex = {}
    if task_name == "multirc":
        # Remove HTML markup.
        input_text = re.sub(r"<br>", " ", input_text)
        input_text = re.sub(r"<(/)?b>", "", input_text)

        # Store the data index in the returned example (used by eval)
        ex["idx/paragraph"] = x["idx"]["paragraph"]
        ex["idx/question"] = x["idx"]["question"]
        ex["idx/answer"] = x["idx"]["answer"]
    else:
        # Store the data index in the returned example (used by eval)
        if id_key:
            ex["idx"] = x[id_key]

    ex["source_text"] = input_text.strip()
    ex["target_text"] = label_name
    if decoder_only:

        # @TODO:
        # 1. evaluation needs padding on left and hence, we need to be consistent with how we pad in training and evaluating
        # 2. we should not have any token after "Answer:", so no padding on right


        input_text = input_text.strip().replace(task_name, "")
        possible_answers = ", ".join(clf_label_names_mapping[task_name])
        input_text = clf_task_description_mapping[task_name] + " " + input_text + " " + f"Select answer from: {possible_answers}. Answer:"
        ex["source_text"] = input_text

    return ex


# Postprocessing functions
def postprocess_summarization(preds, labels, dataset_config_name=None):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def postprocess_classification(preds, labels, dataset_config_name=None):
    pred_ids, label_ids = [], []
    for idx, pred in enumerate(preds):
        pred_id = string_label_to_class_id(
            string_label=pred.lower(),
            label_classes=clf_label_names_mapping[dataset_config_name]
        )
        label_id = string_label_to_class_id(
            string_label=labels[idx].lower(),
            label_classes=clf_label_names_mapping[dataset_config_name]
        )
        pred_ids.append(pred_id)
        label_ids.append(label_id)

    return pred_ids, label_ids

def strip_input_tokens_from_generation(generated_tokens, len_input_wo_class, pad_token_id):
    bsz, seq_len = generated_tokens.shape
    for example in range(bsz):
        len_input = len_input_wo_class[example]
        generated_tokens[example, :len_input] = pad_token_id
    return generated_tokens

def string_label_to_class_id(string_label, label_classes, default=-1):
    """Returns index of string_label in label_classes or default if not found."""
    # source: https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/data/postprocessors.py#L41

    # @TODO: I feel like this is quite stringent. Not changing it because I want to keep it
    # same for t5 and Llama

    if string_label in label_classes:
        return label_classes.index(string_label)  # index, because this is a list

    return default
