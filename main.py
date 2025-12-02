import torch
import numpy as np
import torch.nn as nn
from torch.xpu import device


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size//heads
        
        assert (self.head_dim * heads == embed_size)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)
        
    def forward(self, query, keys, values, mask):
        N = query.shape[0]
        query_len, key_len, value_len = query.shape[1], keys.shape[1], values.shape[1]
        
        #split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.head_dim)
        keys = keys.reshape(N, key_len, self.head_dim)
        queries = query.reshape(N, query_len, self.head_dim) 

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        out = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # out shape: (N, hesds, query_len, key_len)
        
        if mask is not None: #for decoder masked attention
            out = out.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(out/ (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)
        # attention shape : (N, heads, quey_len, key_len)
        # value shape : (N, value_len, heads, head_dim)
        # (N, query_len, value_len, head_dim)
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        attention = self.attention(query, key, value, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            dropout,
            forward_expansion,
            max_len):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embeddings = nn.Embedding(src_vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion)
            for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        out = self.dropout(self.word_embeddings(x) + self.position_embeddings(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out

class DecoderBlock(nn.Module):
    def __init__(self ,embed_size, heads, dropout, forward_expansion, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x,x,x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(query,key,value,src_mask)
        return out

class Decoder(nn.Module):
    def __init__(self,
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            dropout,
            forward_expansion,
            max_len):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embeddings = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, dropout, forward_expansion, device)
             for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        x = self.dropout(self.word_embeddings(x) + self.position_embeddings(positions))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out

class Transformer(nn.Module):
    def __init__(self, src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embed_size = 256,
                 num_layers = 6,
                 forward_expansion = 4,
                 heads = 8,
                 dropout = 0.1,
                 device = "cpu",
                 max_len = 100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, dropout, forward_expansion, max_len)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, device, dropout, forward_expansion, max_len)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask, trg_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


