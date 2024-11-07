import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

    class Embedding(nn.Module):
        def __init__(self, vocab_size, embeddding_dim):
            super().__init__()
            self.embedding_layer = nn.Embedding(vocab_size, embeddding_dim)

        def forward(self, x):
            return self.embedding_layer(x)
        
    class PositionalEncoding(nn.Module):
        def __init__(self, position, d_model):
            super().__init__()
            div_term = torch.pow(10000, -torch.arange(0, d_model, 2).float() / d_model)
            pos = torch.arange(position).unsqueeze(1)
            pe = torch.zeros(position, d_model)
            pe[:, 0::2] = torch.sin(pos*div_term)
            pe[:,1::2] = torch.cos(pos * div_term)
            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + self.pe[:x.size(0),:]
            return x

    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            
            assert d_model%num_heads == 0, "Embedding size must be divisible by number of heads"
            self.d_head = d_model//num_heads
            self.query = nn.Linear(d_model, d_model)
            self.key = nn.Linear(d_model, d_model)
            self.value = nn.Linear(d_model, d_model)
            self.fc_out = nn.Linear(d_model, d_model)

        def forward(self, queries, keys, values, mask):
            N = queries.shape[0]
            Q = self.query(queries)
            K = self.key(keys)
            V = self.value(values)

            Q = Q.view(N, -1, self.num_heads, self.d_head).transpose(1, 2)
            K = K.view(N, -1, self.num_heads, self.d_head).transpose(1, 2)
            V = V.view(N, -1, self.num_heads, self.d_head).transpose(1, 2)

            attention = torch.einsum("nhqd, nhkd->nhqk",[Q,K])/(self.d_head**0.5)
            if mask is not None:
                attention = attention.makes_fill(mask==0, float("-inf"))
            
            out = torch.einsum("nhqk,nhvd->nhqd", [attention, V])
            out = out.transpose(1, 2).reshape(N, -1, self.num_heads * self.d_head)

            return self.fc_out(out)
