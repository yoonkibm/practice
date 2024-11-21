import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, position, num_heads, num_encoder_layers, num_decoder_layers):
        super().__init__()
        self.input_embedding = self.Embedding(vocab_size, d_model)
        self.output_embedding = self.Embedding(vocab_size, d_model)
        self.positional_encoding = self.PositionalEncoding(position, d_model)
        self.encoders = nn.ModuleList([self.Encoder(d_model, num_heads) for _ in range(num_encoder_layers)])
        self.decoders = nn.ModuleList([self.Decoder(d_model, num_heads) for _ in range(num_decoder_layers)])
        self.output_linear = nn.Linear(d_model, vocab_size)
    def forward(self, x1, x2, mask):
        x1 = self.input_embedding(x1)
        x1 = torch.add(x1, self.positional_encoding(x1))
        x2 = self.output_embedding(x2)
        x2 = torch.add(x2, self.positional_encoding(x2))
        for encoder in self.encoders:
            x1 = encoder(x1, mask)
        for decoder in self.decoders:
            x2 = decoder(x1, x2, mask)

        logits = self.output_linear(x2)
        
        return logits

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
            x = x + self.pe[:x.size(1),:]
            return x

    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.d_model = d_model
            self.num_heads = num_heads
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

            attention = torch.matmul(Q, K.transpose(-2,-1))/(self.d_head**0.5)
            if mask is not None:
                mask = mask.unsqueeze(1).unsqueeze(2)
                attention = attention.masked_fill(mask==0, float("-inf"))
            attention = torch.softmax(attention, dim=-1)
            if attention.size(-2) != queries.size(-2) or attention.size(-1) != keys.size(-2):
                raise ValueError(
                    f"Softmax output shape is incorrect. Expected (batch, num_heads, query_len, key_len), "
                    f"but got {attention.size()} with query_len={queries.size(-2)} and key_len={keys.size(-2)}"
        )
            
            out = torch.matmul(attention, V)
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
            self.mha = Transformer.MultiHeadAttention(d_model, num_heads)
            self.mha_residual = Transformer.Residual(d_model)
            self.ffn = Transformer.FeedForward(d_model)
            self.ffn_residual = Transformer.Residual(d_model)

        def forward(self, x, mask):
            attn_out = self.mha(x, x, x, mask)
            attn_out = self.mha_residual(x, attn_out)
            ffn_out = self.ffn(attn_out)
            out = self.ffn_residual(attn_out, ffn_out)
            return out

    class Decoder(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.self_attn = Transformer.MultiHeadAttention(d_model, num_heads)
            self.self_attn_residual = Transformer.Residual(d_model)
            self.cross_attn = Transformer.MultiHeadAttention(d_model, num_heads)
            self.cross_attn_residual = Transformer.Residual(d_model)
            self.ffn = Transformer.FeedForward(d_model)
            self.ffn_residual = Transformer.Residual(d_model)
            self.self_attn_query_layer = nn.Linear(d_model, d_model)

        def forward(self, encoder_out, x, mask):
            masked_attn_out = self.self_attn(x, x, x, mask)
            masked_attn_out = self.self_attn_residual(x, masked_attn_out)
            attn_out = self.cross_attn(masked_attn_out, encoder_out, encoder_out, None)
            attn_out = self.cross_attn_residual(masked_attn_out, attn_out)
            ffn_out = self.ffn(attn_out)
            out = self.ffn_residual(attn_out, ffn_out)

            return out    