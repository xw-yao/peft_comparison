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
    "wiki_summary": ("article", "highlights"),
}


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
