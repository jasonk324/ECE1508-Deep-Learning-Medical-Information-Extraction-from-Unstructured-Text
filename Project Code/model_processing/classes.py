import torch.nn as nn
from torchcrf import CRF
import torch
from torch.utils.data import Dataset

# Data type to insert into BiLSTM_CRF
class NERDataset(Dataset):
    def __init__(self, sentences, labels, word_to_idx, label_to_idx):
        self.sentences = sentences
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):

        """
        When getting an item from the dataset the sentence and its labels are converted to their corresponding IDs and returned as tensors.
        """

        tokens = self.sentences[idx]
        labels = self.labels[idx]

        token_ids = []
        for token in tokens: 
            if token in self.word_to_idx:
                token_ids.append(self.word_to_idx[token])
            else:
                token_ids.append(self.word_to_idx["[UNK]"])

        labels_ids = []
        for label in labels:
            labels_ids.append(self.label_to_idx[label])

        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(labels_ids, dtype=torch.long)

# Base BiLSTM_CRF model to be trained
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=128, hidden_dim=256, num_layers=1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.emission = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def compute_emission(self, input_ids):

        """
        Helpful function to compute emissions score from input IDs
        Remember the model till the linear layer only computes emission scores
        """
        
        embeddings = self.embedding(input_ids)       # (B, T, E)
        lstm_out, _ = self.lstm(embeddings)          # (B, T, H)
        emissions = self.emission(lstm_out)          # (B, T, # of labels)
        return emissions
    
    def forward(self, input_ids, tags=None, mask=None):
        
        """
        Training forward pass
        CRF handles the transition score and will merge with the emission scores to get the final sequence scores
        """

        emissions = self.compute_emission(input_ids)
        loss = -self.crf(emissions, tags, mask=mask, reduction="mean") ## TODO make sure to explain more in report
        return loss
    
    def decode(self, input_ids, mask=None):
        
        """
        Inference decode pass
        CRF handles the transition score and will merge with the emission scores to get the final sequence scores
        """

        emissions = self.compute_emission(input_ids)
        predictions = self.crf.decode(emissions, mask=mask) ## TODO make sure to explain more in report
        return predictions
    
# Exact same model as BiTLSTM_CRF but added Dropout layers
class BiLSTM_CRF_Dropout(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=128, hidden_dim=256, num_layers=1, dropout_rate=0.2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.dropout_in = nn.Dropout(dropout_rate)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout_out = nn.Dropout(dropout_rate)

        self.emission = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def compute_emission(self, input_ids):

        """
        Helpful function to compute emissions score from input IDs
        Remember the model till the linear layer only computes emission scores
        """
        
        x = self.embedding(input_ids)    
        x = self.dropout_in(x)
        lstm_out, _ = self.lstm(x)         
        lstm_out = self.dropout_out(lstm_out)
        emissions = self.emission(lstm_out)        
        return emissions
    
    def forward(self, input_ids, tags=None, mask=None):
        
        """
        Training forward pass
        CRF handles the transition score and will merge with the emission scores to get the final sequence scores
        """

        emissions = self.compute_emission(input_ids)
        loss = -self.crf(emissions, tags, mask=mask, reduction="mean") ## TODO make sure to explain more in report
        return loss
    
    def decode(self, input_ids, mask=None):
        
        """
        Inference decode pass
        CRF handles the transition score and will merge with the emission scores to get the final sequence scores
        """

        emissions = self.compute_emission(input_ids)
        predictions = self.crf.decode(emissions, mask=mask) ## TODO make sure to explain more in report
        return predictions