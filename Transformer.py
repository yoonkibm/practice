import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, position, num_heads):
        super().__init__()
        self.embedding = self.Embedding(vocab_size, d_model)
        self.positional_encoding = self.PositionalEncoding(position, d_model)
        self.encoder = self.Encoder(d_model, num_heads)
        self.decoder = self.Decoder(d_model, num_heads)

    def forward(self, x1, x2, n):
        x1 = self.embedding(x1)
        x1 = torch.add(x1, self.positional_encoding(x1))
        for i in range(n):
            x1 = self.encoder(x1, None)
        for i in range(n):
            x2 = self.decoder(x1, x2, True)
        
        return x2

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
                attention = attention.masked_fill(mask==0, float("-inf"))
            
            out = torch.einsum("nhqk,nhvd->nhqd", [attention, V])
            out = out.transpose(1, 2).reshape(N, -1, self.num_heads * self.d_head)

            return self.fc_out(out)
    
    class Residual(nn.Module):
        def __init__(self,d_model):
            super().__init__()
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x1, x2):
            x = torch.add(x1, x2)
            x = self.norm(x)
            return x

    class FeedForward(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.expand_layer = nn.Linear(d_model, 4*d_model)
            self.compress_layer = nn.Linear(4*d_model, d_model)
            self.gelu = nn.GELU()

        def forward(self, x):
            x = self.expand_layer(x)
            x = self.gelu(x)
            x = self.compress_layer(x)
            return x

    class Encoder(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.mha = self.MultiHeadAttention(d_model, num_heads)
            self.mha_residual = self.Residual(d_model)
            self.ffn = self.FeedForward(d_model)
            self.query_layer = nn.Linear(d_model, d_model)
            self.key_layer = nn.Linear(d_model, d_model)
            self.value_layer = nn.Linear(d_model, d_model)
            self.ffn_residual = self.Residual(d_model)

        def forward(self, x, mask):
            q = self.query_layer(x)
            k = self.key_layer(x)
            v = self.value_layer(x)
            attn_out = self.mha(q, k, v, mask)
            attn_out = self.mha_residual(x, attn_out)
            ffn_out = self.ffn(attn_out)
            out = self.ffn_residual(attn_out, ffn_out)
            return out

    class Decoder(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.self_attn = self.MultiHeadAttention(d_model, num_heads)
            self.self_attn_residual = self.Residual(d_model)
            self.cross_attn = self.MultiHeadAttention(d_model, num_heads)
            self.cross_attn_residual = self.Residual(d_model)
            self.ffn = self.FeedForward(d_model)
            self.ffn_residual = self.Residual(d_model)
            self.query_layer = nn.Linear(d_model, d_model)
            self.key_layer = nn.Linear(d_model, d_model)
            self.value_layer = nn.Linear(d_model, d_model)
            self.cross_attn_key_layer = nn.Linear(d_model, d_model)
            self.cross_attn_value_layer = nn.Linear(d_model, d_model)
            self.self_attn_query_layer = nn.Linear(d_model, d_model)

        def forward(self, encoder_out, x, mask):
            q = self.query_layer(x)
            k = self.key_layer(x)
            v = self.value_layer(x)
            masked_attn_out = self.self_attn(q, k, v, mask)
            masked_attn_out = self.self_attn_residual(x, masked_attn_out)
            encoder_key = self.cross_attn_key_layer(encoder_out)
            encoder_value = self.cross_attn_value_layer(encoder_out)
            masked_attn_out_q = self.self_attn_query_layer(masked_attn_out)
            attn_out = self.cross_attn(masked_attn_out_q, encoder_key, encoder_value, None)
            attn_out = self.cross_attn_residual(masked_attn_out, attn_out)
            ffn_out = self.ffn(attn_out)
            out = self.ffn_residual(attn_out, ffn_out)

            return out    