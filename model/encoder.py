import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_vocab: int, embedding_dim: int, enc_hidden_dim: int, dec_hidden_dim: int, dropout: int):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, enc_hidden_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.rnn(embedded)
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        )
        return output, hidden