# Module 06: Attention Mechanism

> **Level**: Intermediate → Advanced  
> **Duration**: 2–3 weeks  
> **Prerequisites**: Module 05 (RNN/LSTM/GRU)  
> **Goal**: Understand attention before Transformers

---

## Table of Contents

1. [The Attention Motivation](#1-the-attention-motivation)
2. [Seq2Seq Bottleneck Problem](#2-seq2seq-bottleneck-problem)
3. [Bahdanau Attention](#3-bahdanau-attention)
4. [Luong Attention](#4-luong-attention)
5. [Attention Mechanisms Taxonomy](#5-attention-mechanisms-taxonomy)
6. [Self-Attention](#6-self-attention)
7. [Multi-Head Attention (Preview)](#7-multi-head-attention-preview)
8. [Attention Visualizations](#8-attention-visualizations)
9. [Applications Beyond NLP](#9-applications-beyond-nlp)
10. [From Attention to Transformers](#10-from-attention-to-transformers)

---

## 1. The Attention Motivation

### 1.1 Human Attention

**Example**: Reading this paragraph
- You don't process every word equally
- You **attend** to important words
- Context determines what's important

### 1.2 Why Machines Need Attention

**Problem with fixed representations**:
```
"The cat sat on the mat" → [0.2, -0.5, 0.8, ...]  (single vector)
```

**Information bottleneck**: Compress entire sentence into one vector!

**Solution**: Let model **attend** to different parts of input.

---

## 2. Seq2Seq Bottleneck Problem

### 2.1 Vanilla Seq2Seq

**Architecture**:
```
Encoder: x1 → x2 → x3 → context (c)
Decoder: c → y1 → y2 → y3
```

**Problem**: Context $c$ must encode **entire** input sequence.

$$
c = h_n \quad \text{(last hidden state)}
$$

**This fails for**:
- Long sequences
- Complex mappings
- Information gets lost

### 2.2 Performance Degradation

**Observation** (Bahdanau et al., 2015):
- Short sentences (10-20 words): Good performance
- Long sentences (>40 words): Dramatic drop

**Why**: Fixed-size bottleneck cannot store all information.

---

## 3. Bahdanau Attention

### 3.1 Key Idea

**Instead of single context vector**, allow decoder to **look at all encoder states**.

$$
c_t = \sum_{i=1}^{T_x} \alpha_{ti} h_i
$$

Where:
- $c_t$ = Context vector at decoder step $t$
- $h_i$ = Encoder hidden state at position $i$
- $\alpha_{ti}$ = Attention weight (how much to attend to $h_i$ at step $t$)

### 3.2 Attention Weights

**Alignment score**:
$$
e_{ti} = a(s_{t-1}, h_i)
$$

Where $a$ is alignment model (typically small MLP):
$$
a(s, h) = v^T \tanh(W_s s + W_h h)
$$

**Softmax normalization**:
$$
\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{k=1}^{T_x} \exp(e_{tk})}
$$

### 3.3 Decoder Update

**With attention**:
$$
s_t = f(s_{t-1}, y_{t-1}, c_t)
$$

**Output**:
$$
p(y_t | y_{<t}, x) = \text{softmax}(W_o [s_t; c_t] + b_o)
$$

### 3.4 Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W_s = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
    
    def forward(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden: (batch, hidden_size)
        encoder_outputs: (batch, seq_len, hidden_size)
        """
        # Expand decoder hidden to match encoder outputs
        # (batch, 1, hidden_size) → (batch, seq_len, hidden_size)
        decoder_hidden = decoder_hidden.unsqueeze(1)
        decoder_hidden = decoder_hidden.repeat(1, encoder_outputs.size(1), 1)
        
        # Compute alignment scores
        # e = v^T tanh(W_s * s + W_h * h)
        energy = torch.tanh(
            self.W_s(decoder_hidden) + self.W_h(encoder_outputs)
        )
        attention_scores = self.v(energy).squeeze(-1)  # (batch, seq_len)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum of encoder outputs
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            encoder_outputs  # (batch, seq_len, hidden_size)
        ).squeeze(1)  # (batch, hidden_size)
        
        return context, attention_weights

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, input_vocab, output_vocab, hidden_size):
        super().__init__()
        
        # Encoder
        self.encoder_embedding = nn.Embedding(input_vocab, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        
        # Attention
        self.attention = BahdanauAttention(hidden_size * 2)
        
        # Decoder
        self.decoder_embedding = nn.Embedding(output_vocab, hidden_size)
        self.decoder = nn.GRU(hidden_size + hidden_size * 2, hidden_size * 2, batch_first=True)
        
        # Output
        self.fc = nn.Linear(hidden_size * 2, output_vocab)
    
    def forward(self, src, tgt):
        # Encode
        src_emb = self.encoder_embedding(src)
        encoder_outputs, encoder_hidden = self.encoder(src_emb)
        
        # Initialize decoder hidden (use last encoder hidden)
        decoder_hidden = encoder_hidden[-1]  # Take last layer
        
        outputs = []
        attention_weights_list = []
        
        # Decode step by step
        for t in range(tgt.size(1)):
            # Get current target embedding
            tgt_emb = self.decoder_embedding(tgt[:, t:t+1])
            
            # Compute attention context
            context, attention_weights = self.attention(decoder_hidden, encoder_outputs)
            attention_weights_list.append(attention_weights)
            
            # Concatenate context with target embedding
            decoder_input = torch.cat([tgt_emb.squeeze(1), context], dim=1).unsqueeze(1)
            
            # Decoder step
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden.unsqueeze(0))
            decoder_hidden = decoder_hidden.squeeze(0)
            
            # Output projection
            output = self.fc(decoder_output.squeeze(1))
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        attention_weights = torch.stack(attention_weights_list, dim=1)
        
        return outputs, attention_weights
```

---

## 4. Luong Attention

### 4.1 Differences from Bahdanau

| Aspect | Bahdanau | Luong |
|--------|----------|-------|
| **When computed** | Before decoder RNN | After decoder RNN |
| **Decoder input** | Concatenate context + input | Just input |
| **Alignment function** | Additive (concat + MLP) | Multiplicative (dot product) |
| **Performance** | Slightly slower | Faster |

### 4.2 Scoring Functions

**Dot product** (simplest):
$$
\text{score}(h_t, \bar{h}_s) = h_t^T \bar{h}_s
$$

**General**:
$$
\text{score}(h_t, \bar{h}_s) = h_t^T W_a \bar{h}_s
$$

**Concat** (like Bahdanau):
$$
\text{score}(h_t, \bar{h}_s) = v_a^T \tanh(W_a [h_t; \bar{h}_s])
$$

### 4.3 Implementation

```python
class LuongAttention(nn.Module):
    def __init__(self, hidden_size, method='dot'):
        super().__init__()
        self.method = method
        self.hidden_size = hidden_size
        
        if method == 'general':
            self.W_a = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == 'concat':
            self.W_a = nn.Linear(hidden_size * 2, hidden_size)
            self.v_a = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden: (batch, hidden_size)
        encoder_outputs: (batch, seq_len, hidden_size)
        """
        if self.method == 'dot':
            # h_t^T * h_s
            scores = torch.bmm(
                decoder_hidden.unsqueeze(1),  # (batch, 1, hidden)
                encoder_outputs.transpose(1, 2)  # (batch, hidden, seq_len)
            ).squeeze(1)  # (batch, seq_len)
        
        elif self.method == 'general':
            # h_t^T * W_a * h_s
            energy = self.W_a(encoder_outputs)
            scores = torch.bmm(
                decoder_hidden.unsqueeze(1),
                energy.transpose(1, 2)
            ).squeeze(1)
        
        elif self.method == 'concat':
            # v_a^T * tanh(W_a * [h_t; h_s])
            decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
            concat = torch.cat([decoder_hidden, encoder_outputs], dim=2)
            energy = torch.tanh(self.W_a(concat))
            scores = self.v_a(energy).squeeze(-1)
        
        # Attention weights
        attention_weights = F.softmax(scores, dim=1)
        
        # Context vector
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)
        
        return context, attention_weights
```

---

## 5. Attention Mechanisms Taxonomy

### 5.1 By Alignment Function

| Type | Formula | Complexity |
|------|---------|-----------|
| **Dot Product** | $s^T h$ | $O(1)$ |
| **Scaled Dot Product** | $\frac{s^T h}{\sqrt{d_k}}$ | $O(1)$ |
| **Additive** | $v^T \tanh(W_1 s + W_2 h)$ | $O(d)$ |
| **Multiplicative** | $s^T W h$ | $O(d)$ |

### 5.2 By Attention Scope

**Global Attention**: Attend to all encoder positions
$$
\alpha_t = \text{softmax}(\text{score}(h_t, \bar{h}))
$$

**Local Attention**: Attend to window around position $p_t$
$$
\alpha_t = \text{softmax}(\text{score}(h_t, \bar{h}_{p_t - D : p_t + D}))
$$

**Hard Attention**: Select single position (discrete, non-differentiable)
$$
\alpha_t \in \{0, 1\}^{T_x}
$$

**Soft Attention**: Weighted average (continuous, differentiable)
$$
\alpha_t \in [0, 1]^{T_x}, \quad \sum_i \alpha_{ti} = 1
$$

---

## 6. Self-Attention

### 6.1 Motivation

**RNN attention**: Decoder attends to encoder  
**Self-attention**: Sequence attends to **itself**

**Use case**: Capture relationships within a single sequence.

**Example**:
```
"The animal didn't cross the street because it was too tired"
```
What does "it" refer to? → "animal" (not "street")

### 6.2 Self-Attention Mechanism

**Input**: Sequence of vectors $X = [x_1, \ldots, x_n]$

**For each position $i$**:
1. Compute attention to all positions
2. Context vector = weighted sum

$$
\text{Attention}(x_i, X) = \sum_{j=1}^{n} \alpha_{ij} x_j
$$

Where:
$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}
$$

$$
e_{ij} = \text{score}(x_i, x_j)
$$

### 6.3 Query-Key-Value Formulation

**Key insight**: Represent each element with 3 vectors.

**Query (Q)**: What I'm looking for  
**Key (K)**: What I have to offer  
**Value (V)**: What I'll actually contribute

$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$

**Attention**:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

### 6.4 Implementation

```python
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads=1):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim * heads == embed_size, "Embed size must be divisible by heads"
        
        self.W_q = nn.Linear(embed_size, embed_size)
        self.W_k = nn.Linear(embed_size, embed_size)
        self.W_v = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, embed_size)
        mask: (batch, seq_len, seq_len) - optional
        """
        N, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_q(x)  # (N, seq_len, embed_size)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Split into multiple heads
        Q = Q.reshape(N, seq_len, self.heads, self.head_dim).transpose(1, 2)
        K = K.reshape(N, seq_len, self.heads, self.head_dim).transpose(1, 2)
        V = V.reshape(N, seq_len, self.heads, self.head_dim).transpose(1, 2)
        # Now: (N, heads, seq_len, head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # scores: (N, heads, seq_len, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights
        attention = torch.softmax(scores, dim=-1)
        
        # Weighted sum of values
        out = torch.matmul(attention, V)  # (N, heads, seq_len, head_dim)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().reshape(N, seq_len, self.embed_size)
        
        # Final linear layer
        out = self.fc_out(out)
        
        return out, attention
```

### 6.5 Why Scaling by $\sqrt{d_k}$?

**Problem**: For large $d_k$, dot products grow large in magnitude.

**Example**: Two random unit vectors
$$
\mathbb{E}[q \cdot k] = 0, \quad \text{Var}[q \cdot k] = d_k
$$

**Solution**: Scale to maintain unit variance
$$
\frac{q \cdot k}{\sqrt{d_k}}
$$

**Effect**: Prevents softmax from saturating (all weight on one position).

---

## 7. Multi-Head Attention (Preview)

### 7.1 Intuition

**Single attention**: One representation of relevance  
**Multi-head**: Multiple "representation subspaces"

**Analogy**: 
- Head 1: Syntactic relationships (subject-verb)
- Head 2: Semantic relationships (word meaning)
- Head 3: Positional relationships (adjacent words)

### 7.2 Formula

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

Where:
$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

**Parameters**:
- $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$

**Typically**: $d_k = d_v = d_{\text{model}} / h$

---

## 8. Attention Visualizations

### 8.1 Attention Heatmaps

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(attention_weights, source_tokens, target_tokens):
    """
    attention_weights: (target_len, source_len)
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights,
        xticklabels=source_tokens,
        yticklabels=target_tokens,
        cmap='viridis',
        cbar=True
    )
    plt.xlabel('Source')
    plt.ylabel('Target')
    plt.title('Attention Weights')
    plt.show()

# Example usage
source = ['The', 'cat', 'sat', 'on', 'the', 'mat']
target = ['Le', 'chat', 'était', 'assis']
attention = torch.randn(4, 6).softmax(dim=1)

plot_attention(attention.numpy(), source, target)
```

### 8.2 Interpreting Attention

**High attention weight** ($\alpha_{ij} \approx 1$):
- Position $j$ is very relevant for predicting position $i$

**Example** (Translation):
```
Source: "The    cat    sat    on    the    mat"
Target: "Le" → [0.7, 0.2, 0.0, 0.0, 0.1, 0.0]  (attends to "The")
        "chat" → [0.1, 0.8, 0.0, 0.0, 0.1, 0.0]  (attends to "cat")
```

---

## 9. Applications Beyond NLP

### 9.1 Image Captioning

**Encoder**: CNN extracts image features  
**Decoder**: RNN with attention to different image regions

```python
class ImageCaptioningWithAttention(nn.Module):
    def __init__(self, encoder, vocab_size, hidden_size):
        super().__init__()
        self.encoder = encoder  # CNN (ResNet, etc.)
        self.attention = LuongAttention(hidden_size)
        self.decoder = nn.LSTM(hidden_size + embed_dim, hidden_size)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, image, caption):
        # Extract image features
        features = self.encoder(image)  # (batch, num_regions, feature_dim)
        
        # Decode with attention to image regions
        # ... (similar to text attention)
```

### 9.2 Speech Recognition

**Attention over audio frames**:
```
Audio frames → Encoder → Attention ← Decoder → Text
```

### 9.3 Graph Neural Networks

**Attention over neighbors**:
$$
h_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} W^{(l)} h_j^{(l)}\right)
$$

---

## 10. From Attention to Transformers

### 10.1 Key Limitation of RNN + Attention

**Sequential processing**: Still need RNN backbone
- Cannot parallelize across time
- Slow for long sequences

### 10.2 The Breakthrough (2017)

**"Attention is All You Need"** (Vaswani et al.)

**Key insight**: Use **only** attention, no RNN!

**Transformer**:
- Self-attention for encoder
- Self-attention + cross-attention for decoder
- Positional encoding (since no recurrence)
- Fully parallelizable

### 10.3 Evolution Timeline

```
2014: Seq2Seq (Sutskever et al.)
      ↓
2015: Attention for Seq2Seq (Bahdanau et al.)
      ↓
2017: Transformer (Vaswani et al.)
      ↓
2018: BERT, GPT (attention-only architectures)
      ↓
2020+: GPT-3, LLaMA, etc. (scaled Transformers)
```

---

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [Seq2Seq with Bahdanau Attention](notebooks/01_bahdanau_attention.ipynb) | Implement and visualize attention |
| 2 | [Attention Comparison](notebooks/02_attention_comparison.ipynb) | Bahdanau vs Luong |
| 3 | [Self-Attention from Scratch](notebooks/03_self_attention.ipynb) | Build self-attention layer |
| 4 | [Attention Visualization](notebooks/04_attention_viz.ipynb) | Heatmaps and interpretation |

---

## Projects

### Mini Project: Neural Machine Translation with Attention
- Seq2seq with Bahdanau attention
- Train on WMT dataset
- Visualize attention weights
- Compare BLEU scores with/without attention

### Advanced Project: Image Captioning
- ResNet encoder
- LSTM decoder with attention over image regions
- Train on COCO dataset
- Generate captions for new images
- Visualize which regions model attends to

---

## Interview Questions

1. Explain the bottleneck problem in seq2seq and how attention solves it.
2. Walk through the attention mechanism step-by-step with equations.
3. What's the difference between Bahdanau and Luong attention?
4. Explain self-attention. How is it different from seq2seq attention?
5. Why do we scale dot-product attention by $\sqrt{d_k}$?
6. What's the computational complexity of self-attention?
7. Explain Query, Key, and Value in the context of attention.
8. How does multi-head attention differ from single-head?
9. Why did Transformers replace RNNs for most NLP tasks?
10. Can you have attention without RNNs? (Yes → Transformers!)
