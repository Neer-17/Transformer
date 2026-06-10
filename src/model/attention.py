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
    def __init__(self):
        super().__init__()
        self.d_model = 256
        self.num_heads = 8
        self.d_k = self.d_model // self.num_heads
        self.W_q = nn.Linear(self.d_model,self.d_model)
        self.W_k = nn.Linear(self.d_model,self.d_model)
        self.W_v = nn.Linear(self.d_model,self.d_model)
        self.W_o = nn.Linear(self.d_model,self.d_model)
    
    def forward(self,x:torch.Tensor,mask:bool):
        batch_size = x.size(0)
        seq_len = x.size(1)

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(Q.size(0),Q.size(1),self.num_heads,self.d_k)
        K = K.view(K.size(0),K.size(1),self.num_heads,self.d_k)
        V = V.view(V.size(0),V.size(1),self.num_heads,self.d_k)

        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)


        weights,output = scaled_dot_product_attention(Q,K,V,mask=True)

        output = output.transpose(1,2).contiguous()
        output = output.view(batch_size,seq_len,self.d_model)

        final_output = self.W_o(output)
        return final_output
    

            