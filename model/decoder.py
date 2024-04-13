import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(
        self, 
        output_vocab: int, 
        embedding_dim: int, 
        enc_hidden_dim: int, 
        dec_hidden_dim: int, 
        attention: nn.Module, 
        dropout: float
    ):
        super().__init__()
        self.output_vocab = output_vocab
        self.attention = attention
        self.embedding = nn.Embedding(output_vocab, embedding_dim)
        self.rnn = nn.GRU((enc_hidden_dim*2) + embedding_dim, dec_hidden_dim)
        self.fc = nn.Linear((enc_hidden_dim*2) + embedding_dim + dec_hidden_dim, output_vocab)
        self.dropout = dropout

    def forward(self, input: torch.Tensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        attn = self.attention(hidden, encoder_outputs).unqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(attn, encoder_outputs).permute(1, 20, 2)
        output, hidden = self.rnn(torch.cat((embedded, weighted), dim=2), hidden.unsqueeze(0))
        embedded = embedded.squeeze(0)
        output = embedded.squeeze(0)
        weighted = embedded.squeeze(0)
        prediction = self.fc(torch.cat((output, weighted, embedded), dim=1))
        return prediction, hidden.squeeze(0)
