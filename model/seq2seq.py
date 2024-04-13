import random

import torch
from torch import nn

class Seq2Seq(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.cuda.device):
        super().__init__()
        self.encoder = encoder 
        self.decoder = decoder 
        self.device = device
    
    def forward(self, src: torch.Tensor, target: torch.Tensor, tf_ratio: float) -> torch.Tensor:
        target_length = target.shape[0]

        outputs = torch.zeros(target_length, target.shape[1], self.decoder.output_vocab).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        input = target[0, :]

        for t in range(1, target_length):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            input = target[t] if random.random() < tf_ratio else output.argmax(1)

        return outputs