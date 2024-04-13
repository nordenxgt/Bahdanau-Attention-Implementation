import torch
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, enc_hidden_dim: int, dec_hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear((enc_hidden_dim*2) + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)
    
    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        hidden = hidden.unsqueeze(1).repeat(1, encoder_outputs.shape[0], 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)