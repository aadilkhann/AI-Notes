# Contributing to AI Mastery Repository

Thank you for your interest in contributing! This repository is designed as a comprehensive AI learning resource from beginner to PhD level.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Repository Structure](#repository-structure)
- [Content Guidelines](#content-guidelines)
- [Code Standards](#code-standards)
- [Pull Request Process](#pull-request-process)
- [Writing Notebooks](#writing-notebooks)
- [Creating Projects](#creating-projects)

## Code of Conduct

This project adheres to a code of professionalism and respect:

- **Be respectful** of all skill levels
- **Be constructive** in feedback
- **Be patient** with learners
- **Focus on education** over showing off
- **Cite sources** properly
- **No plagiarism** - always attribute original work

## How to Contribute

We welcome contributions in several areas:

### 1. Content Improvements
- **Fix errors** in math, code, or explanations
- **Add clarity** to complex topics
- **Update outdated information** (papers, libraries, best practices)
- **Improve diagrams** and visualizations

### 2. New Content
- **Add notebooks** on missing topics
- **Create mini projects** for hands-on learning
- **Write advanced projects** with production systems
- **Generate architecture diagrams** using Mermaid

### 3. Code Enhancements
- **Optimize implementations** for clarity or performance
- **Add type hints** and docstrings
- **Write tests** for critical code
- **Improve error handling**

### 4. Documentation
- **Expand README** files
- **Add inline comments** for complex algorithms
- **Create tutorials** for specific workflows
- **Write troubleshooting guides**

## Repository Structure

```
AI-Mastery/
├── 00_Math_Foundations/
│   ├── README.md (Theory with LaTeX)
│   ├── notebooks/ (Jupyter notebooks)
│   ├── projects/ (mini_project/ and advanced_project/)
│   └── diagrams/ (Mermaid diagrams)
├── 01_Python_for_AI/
├── ... (modules 02-19)
└── 20_PHD_TRACK/
```

### Key Principles

1. **Self-contained modules**: Each module should be complete
2. **Progressive difficulty**: Beginner → Advanced within each module
3. **Theory + Practice**: READMEs for theory, notebooks for implementation
4. **Real-world focus**: Projects mirror industry requirements

## Content Guidelines

### Theory (README.md)

**Structure**:
```markdown
# Module Name

## Learning Objectives
- Clear, measurable goals

## Prerequisites
- List required prior knowledge

## Core Concepts
### Concept 1
- Intuition first
- Mathematical formulation (LaTeX)
- Code examples
- Visualizations

## Advanced Topics
...

## References
- Research papers (with links)
- Books
- Blog posts
- Video lectures
```

**Writing Style**:
- **Start with intuition** before math
- **Use analogies** for complex concepts
- **Show derivations** step-by-step for key equations
- **Include complexity analysis** (time/space)
- **Provide real-world context** (why does this matter?)

**Math Formatting**:
```latex
$$
\text{Loss} = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)
$$
```

### Notebooks

**Structure**:
1. **Introduction cell** (goals, prerequisites)
2. **Imports** (with explanations if uncommon)
3. **Theory recap** (brief, linking to README)
4. **Implementation** (from scratch)
5. **Framework implementation** (PyTorch/etc)
6. **Experiments** (visualization, comparison)
7. **Exercises** (for learners)
8. **Further reading**

**Code Standards**:
```python
def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
              mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Scaled dot-product attention.
    
    Args:
        Q: Query matrix (batch, seq_len, d_k)
        K: Key matrix (batch, seq_len, d_k)
        V: Value matrix (batch, seq_len, d_v)
        mask: Optional attention mask (batch, seq_len, seq_len)
    
    Returns:
        Output: (batch, seq_len, d_v)
        
    Math:
        Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    
    return output
```

**Best Practices**:
- ✅ Clear docstrings with math formulas
- ✅ Type hints for all functions
- ✅ Inline comments for non-obvious logic
- ✅ Print shapes during debugging
- ✅ Visualize intermediate results
- ✅ Compare with known implementations
- ❌ No magic numbers
- ❌ No unexplained variable names
- ❌ No code without explanation

### Projects

**Mini Projects** (2-3 weeks):
- Implement core algorithm from scratch
- Training on standard dataset
- Basic evaluation
- Clear grading rubric

**Advanced Projects** (4-6 weeks):
- Production-ready system
- Multiple components (data, model, API, monitoring)
- Deployment with Docker/K8s
- Performance optimization
- Real-world datasets
- Full evaluation suite

**Project README Template**:
```markdown
# Project Title

## Overview
Brief description and learning goals

## Architecture
Mermaid diagram showing system components

## Requirements
- Technical requirements
- Hardware requirements

## Implementation Guide

### Week 1: Data Pipeline
...

### Week 2: Model Development
...

## Evaluation
- Metrics to track
- Target performance
- Grading rubric (100 points)

## Bonus Challenges (+20 points)
...

## Resources
...
```

## Code Standards

### Python Style

Follow **PEP 8** with these specifics:

```python
# Good
def scaled_dot_product_attention(query, key, value, mask=None):
    """Compute scaled dot-product attention."""
    pass

# Bad
def sdp_attn(q,k,v,m=None):  # Unclear abbreviations
    pass
```

**Naming Conventions**:
- `snake_case` for functions and variables
- `PascalCase` for classes
- `UPPER_CASE` for constants
- Descriptive names (avoid `x`, `tmp`, `data`)

### Imports

```python
# Standard library
import math
import os
from typing import Optional, Tuple

# Third-party
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Local
from utils import load_data
```

### Documentation

**Module-level**:
```python
"""
Module: transformer.py
Author: Your Name
Date: 2024-01-15

Implements the Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017).
"""
```

**Class-level**:
```python
class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism.
    
    Splits attention into multiple heads to allow the model to jointly
    attend to information from different representation subspaces.
    
    Args:
        d_model: Model dimensionality
        n_heads: Number of attention heads
        dropout: Dropout probability
    
    Reference:
        Vaswani et al. (2017). "Attention Is All You Need"
        https://arxiv.org/abs/1706.03762
    """
```

### Testing

For critical implementations, add tests:

```python
def test_attention_output_shape():
    """Test that attention produces correct output shape."""
    batch_size, seq_len, d_model = 2, 10, 64
    Q = K = V = torch.randn(batch_size, seq_len, d_model)
    
    output = attention(Q, K, V)
    
    assert output.shape == (batch_size, seq_len, d_model)

def test_attention_mask():
    """Test that causal mask prevents future attention."""
    # Implementation...
```

## Pull Request Process

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/AI-Mastery.git
cd AI-Mastery
git remote add upstream https://github.com/ORIGINAL/AI-Mastery.git
```

### 2. Create Branch

```bash
git checkout -b feature/add-flash-attention-notebook
```

**Branch naming**:
- `feature/` - New content
- `fix/` - Bug fixes
- `docs/` - Documentation only
- `refactor/` - Code improvements

### 3. Make Changes

- Follow all style guidelines
- Test your code
- Add/update documentation
- Run notebooks end-to-end

### 4. Commit

```bash
git add path/to/changed/files
git commit -m "Add Flash Attention implementation notebook"
```

**Commit message format**:
```
Add Flash Attention implementation notebook

- Implement tiled attention algorithm
- Add comparison with standard attention
- Include memory profiling examples
- Add references to Dao et al. (2022)

Closes #42
```

### 5. Push and Create PR

```bash
git push origin feature/add-flash-attention-notebook
```

Then create PR on GitHub with:
- **Clear title** describing the change
- **Description** of what was added/changed and why
- **Testing** details
- **Screenshots** if visual changes

### 6. Review Process

- Maintainers will review within 1 week
- Address feedback with new commits
- Once approved, your PR will be merged!

## Writing Notebooks

### Notebook Checklist

- [ ] Title and overview in first cell
- [ ] All dependencies imported with versions
- [ ] Mathematical derivations with LaTeX
- [ ] Code runs from top to bottom without errors
- [ ] Visualizations are clear and labeled
- [ ] Outputs are saved (don't clear before committing)
- [ ] Exercises for learners
- [ ] References section at end

### Example Cell Structure

**Markdown Cell**:
```markdown
## 2.1 Self-Attention Mechanism

**Intuition**: Each token attends to all other tokens to gather context.

**Math**:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

where:
- $Q$ (queries): What we're looking for
- $K$ (keys): What we have
- $V$ (values): What we return
```

**Code Cell**:
```python
# Cell explanation
def self_attention(x, W_Q, W_K, W_V):
    """Self-attention with learned projections."""
    Q = x @ W_Q  # (batch, seq_len, d_k)
    K = x @ W_K
    V = x @ W_V
    
    # Scaled dot-product
    scores = Q @ K.T / math.sqrt(Q.shape[-1])
    attn_weights = F.softmax(scores, dim=-1)
    output = attn_weights @ V
    
    return output, attn_weights

# Test on random data
x = torch.randn(1, 10, 64)
W_Q = W_K = W_V = torch.randn(64, 64)
output, weights = self_attention(x, W_Q, W_K, W_V)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
```

## Creating Projects

### Project Selection Criteria

**Good projects**:
- ✅ Reinforce core module concepts
- ✅ Industry-relevant
- ✅ Clear success metrics
- ✅ Reasonable scope (2-6 weeks)
- ✅ Reusable components

**Avoid**:
- ❌ Toy problems without real applications
- ❌ Overly complex multi-month projects
- ❌ Deprecated techniques
- ❌ Projects requiring expensive compute

### Project Structure

```
mini_project/
├── README.md (Full guide)
├── data/ (Sample data or scripts to download)
├── src/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── notebooks/
│   └── exploration.ipynb
├── tests/
│   └── test_model.py
├── requirements.txt
└── .gitignore
```

## Questions?

- Open an issue for clarifications
- Join discussions
- Check existing notebooks for examples
- Refer to this guide before contributing

## Recognition

All contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in relevant module READMEs
- Acknowledged in release notes

Thank you for helping build a world-class AI education resource! 🚀
