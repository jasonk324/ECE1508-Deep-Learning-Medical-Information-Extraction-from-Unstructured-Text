import os
import torch
from torch.utils.data import DataLoader

from model_processing.data_functions import (
    load_data,
    split_dataset,
    build_vocab,
    build_BIO_label_map,
    ner_collate_fn,
    get_trial_number,
)
from model_processing.classes import NERDataset, BiLSTM_CRF, BiLSTM_CRF_Dropout
from model_processing.model_functions import train, save_training_info

# ------------------------------
# Hyperparameters to tune 

min_freq = 5
batch_size = 64
epochs = 40

embedding_dim = 128
hidden_dim = 256
dropout_rate = 0.2

# Training Variables
lr = 0.001
l1 = True
l1_lambda = 1e-5

early_stop = True
patience = 3
min_delta = 0.3
train_delta = 0.1

special_notes = "Final Test I Hope"

# ------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processed_data_path = "./data/processed/sim_sum_gem3_processed.jsonl"
folder_path = f"./models/trial_{get_trial_number()}"
os.makedirs(folder_path, exist_ok=True)

# -----------------------------

sentences, labels = load_data(processed_data_path)
(train_sentences, train_labels), (val_sentences, val_labels), (test_sentences, test_labels) = split_dataset(sentences, labels)

word_to_idx = build_vocab(train_sentences, min_freq=min_freq)
label_to_idx, idx_to_label = build_BIO_label_map(train_labels)

train_dataset = NERDataset(train_sentences, train_labels, word_to_idx, label_to_idx)
val_dataset = NERDataset(val_sentences, val_labels, word_to_idx, label_to_idx)
test_dataset = NERDataset(test_sentences, test_labels, word_to_idx, label_to_idx)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=ner_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=ner_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=ner_collate_fn)

# model = BiLSTM_CRF(vocab_size=len(word_to_idx), tagset_size=len(label_to_idx), embedding_dim=embedding_dim, hidden_dim=hidden_dim)
model = BiLSTM_CRF_Dropout(vocab_size=len(word_to_idx), tagset_size=len(label_to_idx), embedding_dim=embedding_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate)

train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=epochs,
    device=device,
    save_path=folder_path,
    lr=lr,
    l1=l1,
    l1_lambda=l1_lambda,
    early_stop=early_stop,
    patience=patience,
    min_delta=min_delta,
    train_delta=train_delta,
)

save_training_info(
    save_path=folder_path,
    min_freq=min_freq,
    batch_size=batch_size,
    epochs=epochs,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    lr=lr,
    l1=l1,
    l1_lambda=l1_lambda,
    early_stop=early_stop,
    patience=patience,
    min_delta=min_delta,
    train_delta=train_delta,
    special_notes=special_notes,
)