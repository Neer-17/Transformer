import torch
import torch.nn as nn
from positional_encoding import PE
from encoder import Encoder
from decoder import Decoder
class Transformer(nn.Module):
    def __init__(self,vocab_size,d_model,num_heads,num_layers,d_ff,max_seq_len,dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.pe = PE(d_model,max_seq_len)
        self.encoder = Encoder(d_model,num_heads,d_ff,dropout,num_layers)
        self.decoder = Decoder(d_model,num_heads,d_ff,dropout,num_layers)
        self.vocab = nn.Linear(d_model,vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,src,tgt,src_mask,tgt_mask):
        src_emb = self.embedding(src)
        src_pe = self.pe(src_emb)
        src_drop = self.dropout(src_pe)
        encoder_output = self.encoder(src_drop,src_mask)

        tgt_emb = self.embedding(tgt)
        tgt_pe = self.pe(tgt_emb)
        tgt_drop = self.dropout(tgt_pe)

        decoder_output = self.decoder(tgt_drop,encoder_output,src_mask,tgt_mask)
        final_output = self.vocab(decoder_output)
        return final_output
        