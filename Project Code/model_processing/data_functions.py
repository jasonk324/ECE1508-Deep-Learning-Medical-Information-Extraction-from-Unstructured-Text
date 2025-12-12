import json
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

def get_trial_number(directory="./models"):
    
    """
    Scans the directory "./models" and returns what the next trial number needs to be by incremented the highest value avalible within the existing directory
    """

    pattern = re.compile(r"^trial_(\d+)$")
    max_trial = 0

    for name in os.listdir(directory):
        path = os.path.join(directory, name)

        # Only consider directories
        if os.path.isdir(path):
            match = pattern.match(name)
            if match:
                trial_num = int(match.group(1))
                if trial_num > max_trial:
                    max_trial = trial_num

    return max_trial + 1

def load_data(jsonl_path):

    """
    Load existing data from jsonl_path to process into 2 list
    """

    sentences, labels = [], []   

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            sentences.append(entry["tokens"])
            labels.append(entry["labels"])

    print(f"Loaded {len(sentences)} samples from JSONL.")

    return sentences, labels

def split_dataset(sentences, labels, val_size=0.15, test_size=0.15, random_state=42):

    """"
    Split the dataset into training, validation and test sets.
    """

    sentence_train, sentence_val_test, label_train, label_val_test = train_test_split(sentences, labels, test_size=(val_size + test_size), random_state=random_state)
    sentence_val, sentence_test, label_val, label_test = train_test_split(sentence_val_test, label_val_test, test_size=(test_size / (val_size + test_size)), random_state=random_state)

    print(f"Training set size: {len(sentence_train)}")
    print(f"Validation set size: {len(sentence_val)}") 
    print(f"Test set size: {len(sentence_test)}") 

    return (sentence_train, label_train), (sentence_val, label_val), (sentence_test, label_test)  

def build_vocab(sentences, min_freq=1):

    """ 
    Build vocabulary from sentences with words appearing at least a minimum frequency of times.
    Returns a dictionary mapping words to unique indices.

    [PAD] is used to pad sequences to the same length when training the neural network in batches.
    [UNK] is used to represent words that are not in the vocabulary when inferencing. 

    {'[PAD]': 0, '[UNK]': 1, 'Patient': 2, 'reports': 3, ...}
    """

    counter = Counter()
    for sentence in sentences:
        counter.update(sentence)

    vocab = ["[PAD]", "[UNK]"] + [w for w, c in counter.items() if c >= min_freq]
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    return word_to_idx

def build_BIO_label_map(labels):

    """ 
    Create mappings for each possible BIO label to a unique index and vice versa.
    Returns two dictionaries: label_to_idx and idx_to_label.

    {'B-LOCATION': 0, 'B-MEASUREMENT': 1, 'B-PROBLEM': 2, ...}
    {0: 'B-LOCATION', 1: 'B-MEASUREMENT', 2: 'B-PROBLEM', ...}
    """    

    unique_tags = sorted({tag for seq in labels for tag in seq})
    label_to_idx = {tag: i for i, tag in enumerate(unique_tags)}
    idx_to_label = {i: tag for tag, i in label_to_idx.items()}
    return label_to_idx, idx_to_label

def ner_collate_fn(batch):

    """
    Pads sequences and creates a mask.
    Assumes [PAD] has index 0 in word2idx.

    For input:
    batch = [
        (tensor([10, 20, 30]), tensor([0, 0, 3])),
        (tensor([50, 60]),     tensor([1, 0])),
        (tensor([80]),         tensor([0]))
    ]

    Outputs:
    Tokens padded
    [[10,20,30],
    [50,60, 0],
    [80, 0, 0]]

    Label padded
    [[0,0,3],
    [1,0,0],
    [0,0,0]]

    Mask output - This is to know which tokens are real and which are just padding
    [[T,T,T],
    [T,T,F],
    [T,F,F]]
    """

    token_seqs, label_seqs = zip(*batch) # token_seqs = (seq0_tokens, seq1_tokens, seq2_tokens) and tag_seqs = (seq0_labels, seq1_labels, seq2_labels)
    batch_size = len(token_seqs)
    sequence_lens = [len(seq) for seq in token_seqs]
    max_len = max(sequence_lens) # Want to make all sequences in the batch the same length as the longest one

    tokens_padded = torch.zeros((batch_size, max_len), dtype=torch.long)  # 0 = [PAD]
    labels_padded = torch.zeros((batch_size, max_len), dtype=torch.long)  # pad with 0 (ignored by mask)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    # Iterate through each sequence in the batch and fill in the padded tensors
    for i, (tokens, labels) in enumerate(zip(token_seqs, label_seqs)):
        seq_len = len(tokens)
        tokens_padded[i,:seq_len] = tokens
        labels_padded[i,:seq_len] = labels
        mask[i,:seq_len] = True

    return tokens_padded, labels_padded, mask

def plot_vocab_coverage(denominator_vocabs, numerator_vocabs, titles):

    """
    Plot vocabulary coverage for multiple numerator / denominator pairs.
    Each subplot shows: |numerator ∩ denominator| / |denominator|
    """

    fig, axes = plt.subplots(len(denominator_vocabs), 1, figsize=(10, 1.5 * len(denominator_vocabs)), sharex=False)

    for ax, denom_vocab, numer_vocab, title in zip(axes, denominator_vocabs, numerator_vocabs, titles):

        denom_set = set(denom_vocab)
        numer_set = set(numer_vocab)

        total_vocab = len(denom_set)
        included = len(numer_set & denom_set)
        missing = total_vocab - included

        ax.barh([0], included, color="navy", label="Included")
        ax.barh([0], missing, left=included, color="lightgray", label="Missing")

        ax.set_xlim(0, total_vocab)
        ax.set_yticks([])
        ax.set_title(f"{title}: {included} / {total_vocab} words ({included / total_vocab:.1%})")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

def compute_vocab_coverage(denominator_vocab, numerator_vocab):

    """
    Compute vocabulary coverage of numerator_vocab by denominator_vocab.
    Coverage = |numerator ∩ denominator| / |numerator|
    """

    denom_set = set(denominator_vocab)
    numer_set = set(numerator_vocab)

    total = len(numer_set)
    covered = len(numer_set & denom_set)
    coverage = covered / total

    print(f"Validation vocab coverage: {covered}/{total} tokens ({coverage:.1%})")

def analyze_word_frequencies(sentences, head_n=10, tail_n=10, min_freq=None):
    
    """
    Analyze token frequency statistics in a tokenized text dataset.

    This function computes frequency counts for all tokens in the dataset, prints summary statistics, and returns a frequency-sorted DataFrame.
    It optionally analyzes the effect of applying a minimum frequency cutoff on token coverage.
    """

    all_tokens = [token for sentence in sentences for token in sentence]
    total_tokens = len(all_tokens)
    print(f"Total number of tokens in dataset: {total_tokens}\n")

    counter = Counter(all_tokens)

    df = pd.DataFrame(counter.items(), columns=["word", "frequency"])
    df = df.sort_values(by="frequency", ascending=False).reset_index(drop=True)

    print(f"Top {head_n} most frequent words:")
    print(df.head(head_n))

    print(f"\nBottom {tail_n} least frequent words:")
    print(df.tail(tail_n))

    if min_freq is not None:
        df_kept = df[df["frequency"] >= min_freq]
        df_removed = df[df["frequency"] < min_freq]

        kept_tokens = df_kept["frequency"].sum()
        removed_tokens = df_removed["frequency"].sum()

        print(
            f"\nToken coverage after applying min_freq = {min_freq}:"
            f"\nKept tokens: {kept_tokens} ({kept_tokens / total_tokens:.1%})"
            f"\nRemoved tokens: {removed_tokens} ({removed_tokens / total_tokens:.1%})"
        )

        plt.figure(figsize=(6, 6))

        _, _, autotexts = plt.pie(
            [kept_tokens, removed_tokens],
            labels=["Remaining", "Removed"],
            autopct="%1.1f%%",
            startangle=130,
            colors=["navy", "gray"]
        )

        for autotext in autotexts:
            autotext.set_color("white")

        plt.title(f"Token coverage after minimum frequency = {min_freq}")
        plt.axis("equal")
        plt.show()
    
    return df