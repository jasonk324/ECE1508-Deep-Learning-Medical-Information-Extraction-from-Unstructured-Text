import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

def load_model(model_class, save_path, vocab_size, tagset_size, device, embedding_dim, hidden_dim, dropout=None):

    '''
    This function loads an existing model from a saved path
    '''

    if dropout == None: 
        model = model_class(vocab_size=vocab_size, tagset_size=tagset_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    else:
        model = model_class(vocab_size=vocab_size, tagset_size=tagset_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, dropout_rate=dropout)

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)
    model.eval()

    return model

def evaluate_token_accuracy(model, dataset, device):

    '''
    This function is able to load test data of class NERDataset with a model and output what the accuracy per token is 
    '''

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_correct = 0
    total_tokens = 0

    model.eval()

    with torch.no_grad():
        for tokens, true_labels in loader:
            tokens = tokens.to(device)
            true_labels = true_labels.to(device)

            predictions = model.decode(tokens)

            for pred_seq, gold_seq in zip(predictions, true_labels):
                pred_seq = torch.tensor(pred_seq, device=device)
                gold_seq = gold_seq[: len(pred_seq)]

                total_correct += (pred_seq == gold_seq).sum().item()
                total_tokens += len(pred_seq)

    return total_correct / total_tokens

def show_prediction_example(model, sentences, labels, word_to_idx, idx_to_label, sentence_idx, device):
    
    '''
    Returns a dataframe output if the model was to inference over only a single sample. 

    Each row represents a token in the sample with the true label and predicted label.

    From the test dataset, sentence_idx is the index value of the sample that is selected. 
    '''
    
    tokens = sentences[sentence_idx]
    true_labels = labels[sentence_idx]

    token_ids = torch.tensor([word_to_idx.get(t, word_to_idx["[UNK]"]) for t in tokens], dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_ids = model.decode(token_ids)[0]

    pred_labels = [idx_to_label[i] for i in pred_ids]
    df = pd.DataFrame({"Token": tokens, "True Label": true_labels, "Predicted Label": pred_labels})

    return df

def plot_from_csv(csv_path):

    '''
    When each model was generated the loss curve was saved as well in the form of a CSV.

    The purpose of this function is so be able to plot that csv file if desired.
    '''

    df = pd.read_csv(csv_path)

    epochs = df["epoch"]
    train_losses = df["train_loss"]
    val_losses = df["val_loss"]
    train_accs = df["train_acc"]
    val_accs = df["val_acc"]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Training")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Training")
    plt.legend()

    plt.tight_layout()
    plt.show()
