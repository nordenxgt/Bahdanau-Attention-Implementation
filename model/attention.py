import torch
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, enc_hidden_dim: int, dec_hidden_dim: int):
        super().__init__()
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.attn = nn.Linear((enc_hidden_dim*2) + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Parameter(torch.rand(dec_hidden_dim))
    
    def forward(self, hidden: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        batch_size = encoder_output.shape[1]
        src_len = encoder_output.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_output = encoder_output.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_output), dim=2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention = torch.bmm(v, energy).squeeze(1)
        return F.softmax(attention, dim=1)