import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

#/////////////////////////////////////////////////////////////////////////////////////////////////

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier initialization
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def scaled_dot_product_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
            
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, v)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)
        output = self.combine_heads(attn_output)
        
        return self.out_proj(output)
    
    
#///////////////////////////////////////////////////////////////////////////////////////


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]
    

#//////////////////////////////////////////////////////////////////////////////////////////


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)
    
#///////////////////////////////////////////////////////////////////////////////////////


class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    
#/////////////////////////////////////////////////////////////////////////////////////


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.self_attn(x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        
        return x
    
#//////////////////////////////////////////////////////////////////////////////////////

class Encoder(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_layers: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
#////////////////////////////////////////////////////////////////////////////////////////


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.self_attn(x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Cross-attention with residual connection
        cross_output = self.cross_attn(x, encoder_output)
        x = x + self.dropout(cross_output)
        x = self.norm2(x)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm3(x)
        
        return x

#//////////////////////////////////////////////////////////////////////////////


class Decoder(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_layers: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
    
    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, mask)
        return x
    
#///////////////////////////////////////////////////////////////////////////////////////////

class Transformer(nn.Module):
    def __init__(
        self, 
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder = Encoder(d_model, num_layers, num_heads, d_ff, dropout)
        self.decoder = Decoder(d_model, num_layers, num_heads, d_ff, dropout)
        
    def forward(
        self, 
        src: torch.Tensor, 
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask)
        return decoder_output

