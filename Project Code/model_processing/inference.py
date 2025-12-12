import torch

from model_processing.classes import NERDataset, BiLSTM_CRF_Dropout

from model_processing.data_functions import (
    load_data,
    split_dataset,
    build_vocab,
    build_BIO_label_map
)

from model_processing.inference_functions import (
    load_model,
    evaluate_token_accuracy,
    show_prediction_example,
    plot_from_csv
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processed_data_path = "./data/processed/sim_sum_gem3_processed.jsonl"
trial_version = "trial_18"
model_path = f"./models/{trial_version}/model.pt"
curve_path = f"./models/{trial_version}/training_data.csv"

embedding_dim, hidden_dim = 128, 256

sentences, labels = load_data(processed_data_path)
(train_sentences, train_labels), (val_sentences, val_labels), (test_sentences, test_labels) = split_dataset(sentences, labels)

word_to_idx = build_vocab(train_sentences, min_freq=5)
label_to_idx, idx_to_label = build_BIO_label_map(train_labels)

test_dataset = NERDataset(test_sentences, test_labels, word_to_idx, label_to_idx)

model = load_model(
    BiLSTM_CRF_Dropout,
    save_path=model_path,
    vocab_size=len(word_to_idx),
    tagset_size=len(label_to_idx),
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    device=device,
)

test_acc = evaluate_token_accuracy(model, test_dataset, device)
print(f"Test Token Accuracy: {test_acc:.4f}")

sample_id = 0
sample_df = show_prediction_example(model, test_sentences, test_labels, word_to_idx, idx_to_label, sentence_idx=sample_id, device=device)
print(sample_df)

plot_from_csv(curve_path)