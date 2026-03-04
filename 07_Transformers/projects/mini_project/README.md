# Mini Project: Transformer from Scratch

## Overview

Implement a complete Transformer model from scratch for **machine translation**.

**Task**: English → French translation using Multi30k dataset

**Goal**: Achieve BLEU score >25 using only NumPy and PyTorch primitives (no `nn.Transformer`).

## Learning Objectives

1. Understand self-attention mechanism deeply
2. Implement multi-head attention
3. Build positional encoding
4. Create encoder-decoder architecture
5. Train sequence-to-sequence model
6. Evaluate with BLEU score

## Project Structure

```
mini_project/
├── README.md
├── data/
│   ├── download_data.py
│   └── tokenizer.py
├── model/
│   ├── attention.py           # Self-attention, multi-head
│   ├── encoder.py             # Transformer encoder
│   ├── decoder.py             # Transformer decoder
│   ├── transformer.py         # Full model
│   └── positional_encoding.py
├── training/
│   ├── train.py
│   └── translate.py
└── evaluation/
    ├── bleu.py
    └── visualization.py
```

## Part 1: Self-Attention Implementation (20 points)

### Mathematical Foundation

Given input sequence $X \\in \\mathbb{R}^{n \\times d_{\\text{model}}}$:

$$Q = XW_Q, \\quad K = XW_K, \\quad V = XW_V$$

$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$

```python
# model/attention.py
import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    \"\"\"Scaled Dot-Product Attention.\"\"\"
    
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        \"\"\"
        Args:
            Q: (batch, n_heads, seq_len, d_k)
            K: (batch, n_heads, seq_len, d_k)
            V: (batch, n_heads, seq_len, d_v)
            mask: (batch, 1, seq_len, seq_len) or (batch, 1, 1, seq_len)
        
        Returns:
            output: (batch, n_heads, seq_len, d_v)
            attention_weights: (batch, n_heads, seq_len, seq_len)
        \"\"\"
        d_k = Q.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask (prevent attending to padding/future tokens)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
```

**Deliverables**:
- [ ] Implement scaled dot-product attention
- [ ] Handle masking (padding + causal)
- [ ] Return attention weights for visualization

## Part 2: Multi-Head Attention (20 points)

### Architecture

Multi-head attention runs $h$ attention heads in parallel:

$$\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, ..., \\text{head}_h)W^O$$

where $\\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

```python
class MultiHeadAttention(nn.Module):
    \"\"\"Multi-Head Attention with h parallel attention heads.\"\"\"
    
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, \"d_model must be divisible by n_heads\"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        \"\"\"
        Args:
            Q: (batch, seq_len, d_model)
            K: (batch, seq_len, d_model)
            V: (batch, seq_len, d_model)
            mask: (batch, 1, seq_len, seq_len)
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, n_heads, seq_len, seq_len)
        \"\"\"
        batch_size = Q.size(0)
        
        # Linear projections and split into heads
        # (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_k)
        Q = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        x, attention_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        # (batch, n_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_O(x)
        output = self.dropout(output)
        
        return output, attention_weights
```

**Deliverables**:
- [ ] Implement multi-head attention
- [ ] Verify shapes at each step
- [ ] Test with dummy input

## Part 3: Positional Encoding (15 points)

Transformers have no notion of position. Add positional information:

$$PE_{(pos, 2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d_{\\text{model}}}}\\right)$$
$$PE_{(pos, 2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d_{\\text{model}}}}\\right)$$

```python
# model/positional_encoding.py
class PositionalEncoding(nn.Module):
    \"\"\"Sinusoidal positional encoding.\"\"\"
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        \"\"\"
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding
        \"\"\"
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

**Deliverables**:
- [ ] Implement sinusoidal positional encoding
- [ ] Visualize encoding patterns
- [ ] Verify it captures position information

## Part 4: Encoder Block (15 points)

```python
# model/encoder.py
class FeedForward(nn.Module):
    \"\"\"Position-wise Feed-Forward Network.\"\"\"
    
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    \"\"\"Single Transformer Encoder Layer.\"\"\"
    
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        \"\"\"
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, 1, 1, seq_len) for padding mask
        \"\"\"
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x

class TransformerEncoder(nn.Module):
    \"\"\"Stack of N encoder layers.\"\"\"
    
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, 
                 d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)
    
    def forward(self, x, mask=None):
        # Embedding + positional encoding
        x = self.embedding(x) * self.scale
        x = self.positional_encoding(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return x
```

**Deliverables**:
- [ ] Implement encoder layer
- [ ] Add layer normalization
- [ ] Add residual connections
- [ ] Stack N layers

## Part 5: Decoder Block (15 points)

```python
# model/decoder.py
class DecoderLayer(nn.Module):
    \"\"\"Single Transformer Decoder Layer.\"\"\"
    
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        
        # Masked self-attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Cross-attention (encoder-decoder attention)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        \"\"\"
        Args:
            x: (batch, tgt_len, d_model) - decoder input
            encoder_output: (batch, src_len, d_model)
            src_mask: (batch, 1, 1, src_len) - encoder padding mask
            tgt_mask: (batch, 1, tgt_len, tgt_len) - decoder causal mask
        \"\"\"
        # Masked self-attention
        self_attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))
        
        # Cross-attention
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x
```

**Deliverables**:
- [ ] Implement decoder layer
- [ ] Masked self-attention (causal mask)
- [ ] Cross-attention to encoder
- [ ] Stack N decoder layers

## Part 6: Complete Transformer (15 points)

```python
# model/transformer.py
class Transformer(nn.Module):
    \"\"\"Complete Transformer for machine translation.\"\"\"
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, 
                 n_heads=8, n_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout
        )
        
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout
        )
        
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode source
        encoder_output = self.encoder(src, src_mask)
        
        # Decode target
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        return logits
    
    def generate(self, src, src_mask, max_len=50, start_token=2, end_token=3):
        \"\"\"Autoregressive generation for translation.\"\"\"
        self.eval()
        with torch.no_grad():
            # Encode source once
            encoder_output = self.encoder(src, src_mask)
            
            # Start with <BOS> token
            tgt = torch.full((src.size(0), 1), start_token, device=src.device)
            
            for _ in range(max_len):
                # Create causal mask
                tgt_mask = self.create_causal_mask(tgt.size(1), src.device)
                
                # Decode
                decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
                
                # Get next token
                logits = self.output_projection(decoder_output[:, -1, :])
                next_token = logits.argmax(dim=-1, keepdim=True)
                
                # Append to sequence
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # Stop if <EOS>
                if (next_token == end_token).all():
                    break
            
            return tgt
    
    @staticmethod
    def create_causal_mask(size, device):
        \"\"\"Create causal mask to prevent attending to future tokens.\"\"\"
        mask = torch.tril(torch.ones(size, size, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)
```

**Deliverables**:
- [ ] Complete transformer model
- [ ] Training forward pass
- [ ] Autoregressive generation
- [ ] Causal masking

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| Self-Attention | 20 | Correct implementation, masking |
| Multi-Head Attention | 20 | Parallel heads, concatenation |
| Positional Encoding | 15 | Sinusoidal patterns, visualization |
| Encoder | 15 | Layer norm, residual, stacking |
| Decoder | 15 | Masked self-attn, cross-attn |
| Full Model | 15 | Training + inference, BLEU >25 |
| **Total** | **100** | |

## Expected Results

With proper implementation:
- **Training loss**: <2.0 after 20 epochs
- **BLEU score**: 25-30 on Multi30k test set
- **Training time**: ~30 minutes on GPU

## Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

Good luck building the architecture that revolutionized AI!
