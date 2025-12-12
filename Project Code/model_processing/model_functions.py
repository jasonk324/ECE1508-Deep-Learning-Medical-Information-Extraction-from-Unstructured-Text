import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import copy

def validate(model, dataloader, device):

    """
    Calculate validation loss and accuracy of the model
    """

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    with torch.no_grad():
        for tokens, labels, mask in dataloader:
            tokens = tokens.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            loss = model(tokens, labels, mask)
            total_loss += loss.item()

            predictions = model.decode(tokens, mask) # Decode predictions

            # Convert predictions (list of lists) to a padded tensor
            pred_tensor = torch.zeros_like(labels)
            for i, seq in enumerate(predictions):
                pred_tensor[i, :len(seq)] = torch.tensor(seq, device=device)

            # Compute accuracy: only compare masked (non-pad) tokens
            correct = (pred_tensor == labels) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens

    return avg_loss, accuracy

def save_plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):

    """
    Plot the training curve and save the data
    """

    epochs = range(1, len(train_losses) + 1)

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

    fig_path = save_path + "/training_plot.png"
    plt.savefig(fig_path)
    print(f"Saved plot to: {fig_path}")
    plt.show()

    csv_path = save_path + "/training_data.csv"
    df = pd.DataFrame({"epoch": list(epochs), "train_loss": train_losses, "val_loss": val_losses, "train_acc": train_accs, "val_acc": val_accs})    
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")

def save_model(model, save_path):
    
    """
    Save the model's state dictionary to a file.
    """
    
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to: {save_path}")


def train(model, train_loader, val_loader, epochs, device, save_path, lr=0.001, l1=False, l1_lambda=1e-5, early_stop=False, patience=5, min_delta=0.3, train_delta=0.1):
    
    """
    Traing the model using early stop or L1 optionally
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    best_val_loss = float("inf")
    best_train_loss = float("inf")
    epochs_overfitting = 0
    best_model_state = None

    model.to(device)

    for epoch in range(1, epochs + 1):

        model.train()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0

        print(f"\n===== Epoch {epoch}/{epochs} =====")

        progress_bar = tqdm(train_loader, desc="Training", unit="batch")

        for tokens, labels, mask in progress_bar:
            tokens = tokens.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            loss = model(tokens, labels, mask)

            # Apply L1 if L1 = True
            if l1:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_norm

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            with torch.no_grad():
                predictions = model.decode(tokens, mask)

            pred_tensor = torch.zeros_like(labels)
            for i, seq in enumerate(predictions):
                pred_tensor[i, :len(seq)] = torch.tensor(seq, device=device)

            correct = (pred_tensor == labels) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

            progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = total_correct / total_tokens

        avg_val_loss, avg_val_acc = validate(model, val_loader, device)

        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)

        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        print(f"Val   Loss: {avg_val_loss:.4f}, Val   Acc: {avg_val_acc:.4f}")

        # Early stop code
        if early_stop:
            val_improved = avg_val_loss < best_val_loss - min_delta
            train_improved = avg_train_loss < best_train_loss - train_delta

            if val_improved:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_overfitting = 0
                print("Validation loss improved — model checkpointed")

            else:
                if train_improved:
                    epochs_overfitting += 1
                    print(f"Validation plateau + training improving ({epochs_overfitting}/{patience})")
                else:
                    epochs_overfitting = 0

            best_train_loss = min(best_train_loss, avg_train_loss)

            if epochs_overfitting >= patience:
                print("\nEarly stopping: overfitting detected")
                break

    # Return model if early stop
    if early_stop and best_model_state is not None:
        model.load_state_dict(best_model_state)

    save_model(model, save_path + "/model.pt")
    save_plot_training_curves(
        train_losses, val_losses, train_accs, val_accs, save_path
    )

    return train_losses, val_losses, train_accs, val_accs

def save_training_info(save_path, min_freq, batch_size, epochs, embedding_dim, hidden_dim, lr, l1=False, l1_lambda=1e-5, early_stop=False, patience=5, min_delta=0.3, train_delta=0.1, special_notes=""):

    """
    Saves training configuration values into training_info.txt inside the save_path folder generated for this model trained
    """

    file_path = save_path + "/training_info.txt"

    with open(file_path, "w") as f:
        f.write("=== Training Configuration ===\n\n")

        f.write("Data / Model Parameters:\n")
        f.write(f"min_freq:        {min_freq}\n")
        f.write(f"batch_size:      {batch_size}\n")
        f.write(f"epochs:          {epochs}\n")
        f.write(f"embedding_dim:   {embedding_dim}\n")
        f.write(f"hidden_dim:      {hidden_dim}\n")

        f.write("\nOptimization Parameters:\n")
        f.write(f"learning_rate:   {lr}\n")

        f.write("\nRegularization / Training Strategy:\n")
        f.write(f"L1 regularization enabled: {l1}\n")
        if l1:
            f.write(f"L1 lambda:       {l1_lambda}\n")

        f.write(f"Early stopping enabled:    {early_stop}\n")
        if early_stop:
            f.write(f"patience:        {patience}\n")
            f.write(f"min_delta:      {min_delta}\n")
            f.write(f"train_delta:    {train_delta}\n")

        f.write("\n=== Notes ===\n")
        f.write(special_notes if special_notes else "None")

    print(f"Saved training info to: {file_path}")
    