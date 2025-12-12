import pandas as pd
import os
import ollama
import re
import json
import os

def extract_metadata(row, features):

    """
    Obtain metadata from a dataset row to be put into dictionary format to be inserted into the LLM agent for labelling.
    """

    metadata = {}
    for feature in features:
        metadata[feature] = row[feature]
        
    return metadata

def filter_text(text):

    """
    Filter SimSUM raw data  to remove undesired characters before inserted into the LLM.
    """

    text = re.sub(r"\*\*.*?\*\*", "", text, flags=re.DOTALL)  
    text = text.replace("\n", "")                         
    return text

def llm(system_prompt, input_prompt, ollama_model = "gemma3:12b"):

    response = ollama.chat(
        model=ollama_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_prompt}
        ]
    )

    return response["message"]["content"]

def validate_bio_format(text):

    """
    Validate an LLM output to see if BIO labels are of the correct formatting and only has the correct label. 

    If it is then return True, otherwise false.
    """

    VALID_LABELS = {
        "O",
        "B-PROBLEM", "I-PROBLEM",
        "B-TREATMENT", "I-TREATMENT",
        "B-LOCATION", "I-LOCATION",
        "B-MEASUREMENT", "I-MEASUREMENT",
        "B-TEST", "I-TEST"
    }
    segments = text.strip().split()

    for idx, seg in enumerate(segments):

        if "<" not in seg or ">" not in seg:
            print(f"[ERROR] Segment {idx}: '{seg}' is missing '<' or '>'")
            return False

        if not seg.endswith(">"):
            print(f"[ERROR] Segment {idx}: '{seg}' does not end with '>'")
            return False

        try:
            token, label_with_brackets = seg.split("<", 1)
        except ValueError:
            print(f"[ERROR] Segment {idx}: Could not split '{seg}' into token and label")
            return False

        if token == "":
            print(f"[ERROR] Segment {idx}: Empty token before '<' in '{seg}'")
            return False

        label = label_with_brackets[:-1]  

        if label not in VALID_LABELS:
            print(f"[ERROR] Segment {idx}: Invalid label '{label}' in segment '{seg}'")
            print("Allowed labels:", VALID_LABELS)
            return False

    return True


def parse_bio_string(text):

    """
    Parce a string that has been formatted as: token<LABEL> token<LABEL> ...
    Then return the tokenized list of words and labels.
    """

    tokens = []
    labels = []

    segments = text.strip().split()

    for seg in segments:

        token, label_with_brackets = seg.split("<", 1)
        label = label_with_brackets[:-1]  

        tokens.append(token)
        labels.append(label)

    return tokens, labels

def save_row_to_jsonl(path, index, tokens, labels):

    """
    Saves a processed row to a specified JSONL file.
    The formating will be {"index": idx, "tokens": [...], "labels": [...]}
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    entry = {
        "index": index,
        "tokens": tokens,
        "labels": labels
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")