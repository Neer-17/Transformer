import torch 
import torch.nn as nn
from attention import MultiHeadAttention,scaled_dot_product_attention
from encoder import FeedForward

class DecoderBlock(nn.Module):
    def __init__(self,d_model:int,num_heads:int,d_ff:int,dropout:float):
        super().__init__()
        self.MaskedSelfAttention = MultiHeadAttention(d_model,num_heads)
        self.CrossAttention = MultiHeadAttention(d_model,num_heads)
        self.feedforward = FeedForward(d_model,d_ff,dropout)
       
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self,x:torch.Tensor,encoder_output:torch.Tensor,src_mask=None,tgt_mask=True):
        mask_att_weights,mask_att_output = self.MaskedSelfAttention(x,x,x,tgt_mask)

        dr_output1 = self.drop(mask_att_output)
        res_output1 = x + dr_output1
        norm_output1 = self.norm1(res_output1)

        cross_weights,cross_att_output = self.CrossAttention( norm_output1,encoder_output,encoder_output,src_mask)
        dr_output2 = self.drop(cross_att_output)
        res_output2 = norm_output1 + dr_output2
        norm_ouput2 = self.norm2(res_output2)

        ff_ouput = self.feedforward(norm_ouput2)
        dr_output3 = self.drop(ff_ouput)
        res_output3 = norm_ouput2 + dr_output3
        norm_ouput3 = self.norm3(res_output3)

        return norm_ouput3
    
class Decoder(nn.Module):
    def __init__(self,d_model:int,num_heads:int,d_ff:int,dropout:float,num_layers:int):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(d_model=d_model,num_heads=num_heads,d_ff=d_ff,dropout=dropout)for i in range(num_layers)])
    def forward(self,x:torch.Tensor,encoder_output:torch.Tensor,src_mask:bool,tgt_mask:bool):
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,tgt_mask)
        return x
        
        