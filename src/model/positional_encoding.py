import torch
import torch.nn as nn
import math
class PE(nn.Module):
    def __init__(self,d_model:int, max_seq_len:int):
        super().__init__()
        pe_matrix = torch.zeros(max_seq_len,d_model)
        position = torch.arange(max_seq_len).unsqueeze(1).float()
        two_i = torch.arange(0,d_model,2)
        div_term = torch.exp((two_i/d_model) * (-math.log(10000.0)))
        even_val = torch.sin(position*div_term)
        odd_val = torch.cos(position*div_term)
        pe_matrix[:,::2] = even_val
        pe_matrix[:,1::2] = odd_val
        pe_matrix = pe_matrix.unsqueeze(0)
        self.register_buffer('pe_matrix',pe_matrix)
    
    def forward(self, x:torch.Tensor):
        seq_len = x.size(1)
        encodings = self.pe_matrix[:,:seq_len,:]
        return torch.add(x,encodings)