import torch
import torch.nn as nn
from attention import MultiHeadAttention
class FeedForward(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dropout:float):
        super().__init__()
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff,d_model)
        )
    
    def forward(self,x:torch.Tensor):
        output = self.linear_relu_stack(x)
        return output

class EncoderBlock(nn.Module):
    def __init__(self,d_model:int,num_heads:int,d_ff:int,dropout:float):
        super().__init__()

        self.attention = MultiHeadAttention(d_model=d_model,num_heads=num_heads)
        self.feed_forward = FeedForward(d_model=d_model,d_ff=d_ff,dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x:torch.Tensor,mask:bool):

        att_weights,att_output = self.attention(x,x,x,mask=mask)
        dr_output1 = self.dropout(att_output)
        res_output1 = x + dr_output1
        norm_output1 = self.norm1(res_output1)

        ff_output = self.feed_forward(norm_output1)
        dr_output2 = self.dropout(ff_output)
        res_output2 = norm_output1 + dr_output2
        norm_output2 = self.norm2(res_output2)

        final_output = norm_output2
        return att_weights,final_output
    
class Encoder(nn.Module):
    def __init__(self,d_model:int,num_heads:int,d_ff:int,dropout:float,num_layers:int):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(d_model=d_model,num_heads=num_heads,d_ff=d_ff,dropout=dropout)for i in range(num_layers)])
    def forward(self,x:torch.Tensor,mask:bool):
        for layer in self.layers:
            att_weights,x = layer(x,mask)
        return x
        