import torch
import torch.nn as  nn
def scaled_dot_product_attention(query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, mask:bool)->tuple[torch.Tensor,torch.Tensor]:
    scores = query @ torch.transpose(key,-1,-2)
    scaled_scores = scores/(key.shape[-1]**(1/2))
    if mask:
        mask_tensor = torch.full((1,1,query.shape[-2],query.shape[-2]),float('-inf'))
        mask_tensor = torch.triu(mask_tensor,diagonal=1)
        masked_scores = scaled_scores + mask_tensor
        softmax_score = torch.softmax(masked_scores,-1)
    else:
        softmax_score = torch.softmax(scaled_scores,-1)
    return softmax_score, softmax_score @ value


class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int,num_heads:int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        self.W_q = nn.Linear(self.d_model,self.d_model)
        self.W_k = nn.Linear(self.d_model,self.d_model)
        self.W_v = nn.Linear(self.d_model,self.d_model)
        self.W_o = nn.Linear(self.d_model,self.d_model)
    
    def forward(self,query:torch.Tensor,key:torch.Tensor,value:torch.Tensor,mask:bool):
        batch_size = query.size(0)
        seq_len = query.size(1)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(batch_size,seq_len,self.num_heads,self.d_k)
        K = K.view(batch_size,seq_len,self.num_heads,self.d_k)
        V = V.view(batch_size,seq_len,self.num_heads,self.d_k)

        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)


        weights,output = scaled_dot_product_attention(Q,K,V,mask=mask)

        output = output.transpose(1,2).contiguous()
        output = output.view(batch_size,seq_len,self.d_model)

        final_output = self.W_o(output)
        return weights,final_output
    

            