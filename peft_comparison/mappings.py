# default parameters are selected to match Adapters default parameters
hf_adapter_config_string_to_peft_args = {
    "hf_lora": {
        "r": 8,
        "lora_alpha": 8,
        "target_modules": ["q", "v"],
    },
    "hf_lora_all": {
        "r": 8,
        "lora_alpha": 8,
        "target_modules": ["k", "q", "v", "o", "wi", "wo"],  # assumes T5 model
    },
    "hf_krona": {
        "target_modules": ["q", "v"],
    },
    "hf_loha": {
        "target_modules": ["q", "v"],
    }
}

# @TODO: we might need to create a separate mapping for Llama
# - if may be beneficial to change "choice1" and "choice2" to "A" and "B"
summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "EdinburghNLP/xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}

# @TODO: copa keys should include "choice2" and "question", check with Vlad
task_to_keys = {
    # acceptability kind of tasks
    "cola": ("sentence", None),

    # entailment kind of tasks
    "mrpc": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "axb": ("sentence1", "sentence2"),

    # QA kind of tasks
    "boolq": ("passage", "question"),
    "multirc": ("paragraph", "question"),

    # NLI kind of tasks
    "mnli": ("premise", "hypothesis"),
    "rte": ("premise", "hypothesis"),
    "axg": ("premise", "hypothesis"),
    "cb": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "copa": ("premise", "question", "choice1", "choice2"),

    # others
    "qqp": ("question1", "question2"),
    "sst2": ("sentence", None),

}


clf_label_names_mapping = {
    # taken from,
    # glue: https://github.com/tensorflow/datasets/blob/1e12611a0be5f4753d271d7eb1dde15eb8f0185c/docs/community_catalog/huggingface/glue.md
    # super_glue: https://github.com/tensorflow/datasets/blob/1e12611a0be5f4753d271d7eb1dde15eb8f0185c/docs/community_catalog/huggingface/super_glue.md
    "cola": [
        "unacceptable",
        "acceptable"
    ],
    "mnli": [
        "entailment",
        "neutral",
        "contradiction"
    ],
    "mrpc": [
        "not_equivalent",
        "equivalent"
    ],
    "qnli": [
        "entailment",
        "not_entailment"
    ],
    "qqp": [
        "not_duplicate",
        "duplicate"
    ],
    "rte": [
        "entailment",
        "not_entailment"
    ],
    "sst2": [
        "negative",
        "positive"
    ],
    "stsb": None,
    "ax": [
        "entailment",
        "neutral",
        "contradiction"
    ],
    "wnli": [
        "not_entailment",
        "entailment"
    ],
    "boolq": [
        "false",
        "true"
    ],
    "cb": [
        "entailment",
        "contradiction",
        "neutral"
    ],
    "copa": [
        "choice1",
        "choice2"
    ],
    "multirc": [
        "false",
        "true"
    ],
    "record": None,
    "wic": [
        "false",
        "true"
    ],
    "wsc": [
        "false",
        "true"
    ],
    "wsc.fixed": [
        "false",
        "true"
    ],
    "axb": [
        "entailment",
        "not_entailment"
    ],
    "axg": [
        "entailment",
        "not_entailment"
    ],
}

clf_task_description_mapping = {
    "rte": "Given a premise, identify if the hypothesis entails premise or not.",
    "boolq": "Given a passage and a yes/no question, identify if the answer is \"yes\" or \"no\".",
    "cb": "Given a premise, identify if the hypothesis entails, contradicts or is neutral to the premise.",
    "copa": "Given a premise, a question (cause/effect) and two alternative choices, identify plausible answer from the alternative choices.",
    #"sst2": ,
    #"stsb": ,
    #"ax": ,
    #"wnli": ,
    #"cola": ,
    #"mnli": ,
    #"mrpc": ,
    #"qnli": ,
    #"qqp": ,
    #"multirc": ,
    #"record":,
    #"wic": ,
    #"wsc": ,
    #"wsc.fixed": ,
    #"axb": ,
    #"axg": ,
}