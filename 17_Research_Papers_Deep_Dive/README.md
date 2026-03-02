# Module 17: Research Papers Deep Dive

> **Level**: Advanced to Research  
> **Duration**: 6–8 weeks  
> **Prerequisites**: Modules 07-16  
> **Goal**: Read, understand, and implement seminal AI research papers

---

## Table of Contents

1. [How to Read Research Papers](#1-how-to-read-research-papers)
2. [Foundational Papers (2012-2017)](#2-foundational-papers-2012-2017)
3. [Transformer Era (2017-2020)](#3-transformer-era-2017-2020)
4. [Large Language Models (2020-2023)](#4-large-language-models-2020-2023)
5. [Efficient Training & Inference (2020-2024)](#5-efficient-training--inference-2020-2024)
6. [Alignment & Safety (2022-2024)](#6-alignment--safety-2022-2024)
7. [Multimodal Models (2021-2024)](#7-multimodal-models-2021-2024)
8. [Agents & Tool Use (2022-2024)](#8-agents--tool-use-2022-2024)
9. [Recent Breakthroughs (2024-2026)](#9-recent-breakthroughs-2024-2026)
10. [How to Stay Current](#10-how-to-stay-current)

---

## 1. How to Read Research Papers

### 1.1 Three-Pass Method (Keshav, 2007)

**First Pass** (5-10 minutes):
- Read title, abstract, introduction, conclusion
- Look at figures and section headings
- **Goal**: Get the big picture

**Second Pass** (1 hour):
- Read carefully, but skip proofs/math details
- Note key points, make margin notes
- Look at references for related work
- **Goal**: Understand main contributions

**Third Pass** (4-5 hours):
- Virtually re-implement the paper
- Challenge every assumption
- Think about how you'd do it differently
- **Goal**: Deep understanding

### 1.2 Questions to Ask

**Motivation**:
- What problem does this solve?
- Why is it important?
- What are limitations of previous approaches?

**Method**:
- What's the key innovation?
- What are the technical contributions?
- What are the design choices?

**Evaluation**:
- What datasets/benchmarks are used?
- What are the main results?
- How does it compare to baselines?
- Are ablation studies convincing?

**Impact**:
- What are the broader implications?
- What future work does this enable?
- What are potential risks/limitations?

### 1.3 Paper Reading Template

```markdown
# [Paper Title]

**Authors**: [Names]  
**Venue**: [Conference/Journal, Year]  
**Link**: [ArXiv/URL]

## TL;DR
One-sentence summary

## Problem
What problem does this paper address?

## Key Contributions
1. ...
2. ...

## Method
High-level description of approach

## Results
Main experimental findings

## Strengths
- ...

## Weaknesses
- ...

## Implementation Notes
Key details for reproducing

## Future Directions
What's next?

## Related Papers
- [Paper A]
- [Paper B]
```

---

## 2. Foundational Papers (2012-2017)

### 2.1 AlexNet (2012)

**"ImageNet Classification with Deep Convolutional Neural Networks"**  
Krizhevsky, Sutskever, Hinton | NeurIPS 2012

**Key Contributions**:
- First deep CNN to win ImageNet (top-5 error: 15.3%)
- Popularized ReLU activation
- Data augmentation (crops, flips, color jittering)
- Dropout for regularization
- Multi-GPU training

**Architecture**:
```
Input (224x224x3)
  → Conv1 (11x11, stride 4) → ReLU → MaxPool
  → Conv2 (5x5) → ReLU → MaxPool
  → Conv3-5 (3x3) → ReLU
  → FC6-7 (4096) → Dropout → ReLU
  → FC8 (1000) → Softmax
```

**Impact**: Sparked the deep learning revolution.

**Implementation**: [notebooks/01_alexnet.ipynb](notebooks/01_alexnet.ipynb)

---

### 2.2 Dropout (2014)

**"Dropout: A Simple Way to Prevent Neural Networks from Overfitting"**  
Srivastava et al. | JMLR 2014

**Key Idea**: Randomly drop neurons during training with probability $p$.

**Mathematics**:
$$
h = \text{Dropout}(x, p) = \begin{cases}
\frac{x}{1-p} \cdot \text{Bernoulli}(1-p) & \text{training} \\
x & \text{inference}
\end{cases}
$$

**Why it works**: Ensemble of $2^n$ sub-networks.

---

### 2.3 Batch Normalization (2015)

**"Batch Normalization: Accelerating Deep Network Training"**  
Ioffe, Szegedy | ICML 2015

**Key Idea**: Normalize activations within mini-batch.

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

$$
y_i = \gamma \hat{x}_i + \beta
$$

**Benefits**:
- Faster training (higher learning rates)
- Reduces internal covariate shift
- Acts as regularization

---

### 2.4 ResNet (2015)

**"Deep Residual Learning for Image Recognition"**  
He et al. | CVPR 2016

**Key Contribution**: Residual connections enable training 152+ layer networks.

**Residual Block**:
$$
y = F(x, \{W_i\}) + x
$$

**Why it works**: Identity mapping allows gradients to flow directly.

**Impact**: Winner of ImageNet 2015 (3.6% top-5 error).

**Variants**: ResNet-50, ResNet-101, ResNet-152

---

### 2.5 Seq2Seq (2014)

**"Sequence to Sequence Learning with Neural Networks"**  
Sutskever, Vinyals, Le | NeurIPS 2014

**Architecture**:
```
Encoder: RNN(input) → context vector
Decoder: RNN(context vector) → output
```

**Key Insight**: Fixed-size vector can represent variable-length sequences.

**Applications**: Machine translation, summarization, chatbots.

---

### 2.6 Attention is All You Need (2017)

**Vaswani et al. | NeurIPS 2017**

**Revolutionary**: Replaced RNNs with pure attention mechanism.

**Key Contributions**:
1. **Scaled Dot-Product Attention**:
   $$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

2. **Multi-Head Attention**: Parallel attention in different subspaces

3. **Positional Encoding**: Sinusoidal functions for position info

4. **Architecture**: 6-layer encoder-decoder with feedforward networks

**Results**: State-of-the-art on WMT translation (28.4 BLEU on En-De).

**Impact**: Foundation for all modern LLMs.

**Deep Dive**: [notebooks/02_transformer_from_scratch.ipynb](notebooks/02_transformer_from_scratch.ipynb)

---

## 3. Transformer Era (2017-2020)

### 3.1 GPT (2018)

**"Improving Language Understanding by Generative Pre-Training"**  
Radford et al. | OpenAI

**Key Contributions**:
- Unsupervised pre-training + supervised fine-tuning
- Decoder-only Transformer
- Demonstrated transfer learning for NLP

**Architecture**: 12-layer Transformer decoder (117M params)

**Training**: BooksCorpus (7,000 unpublished books)

**Results**: SOTA on 9 out of 12 tasks.

---

### 3.2 BERT (2018)

**"BERT: Pre-training of Deep Bidirectional Transformers"**  
Devlin et al. | NAACL 2019

**Key Innovations**:

1. **Masked Language Modeling (MLM)**:
   - Randomly mask 15% of tokens
   - Predict masked tokens using bidirectional context
   
   ```
   Input:  "The [MASK] sat on the mat"
   Target: "cat"
   ```

2. **Next Sentence Prediction (NSP)**:
   - Predict if sentence B follows sentence A

**Architecture**: 12-layer encoder (BERT-base: 110M params)

**Pre-training**: BooksCorpus + Wikipedia (3.3B words)

**Results**: SOTA on 11 NLP tasks (GLUE, SQuAD, etc.)

**Impact**: Showed bidirectional pre-training is powerful.

**Comparison**:
| Model | Architecture | Pre-training | Use Case |
|-------|-------------|--------------|----------|
| GPT | Decoder-only | Autoregressive | Generation |
| BERT | Encoder-only | Masked LM | Understanding |

---

### 3.3 GPT-2 (2019)

**"Language Models are Unsupervised Multitask Learners"**  
Radford et al. | OpenAI

**Key Contributions**:
- Scaled to 1.5B parameters
- Demonstrated zero-shot learning
- Trained on WebText (40GB of text)

**Controversy**: Initially not released due to safety concerns.

**Key Finding**: Larger models exhibit emergent capabilities.

---

### 3.4 XLNet (2019)

**"XLNet: Generalized Autoregressive Pretraining"**  
Yang et al. | NeurIPS 2019

**Key Innovation**: Permutation Language Modeling
- Best of both worlds: autoregressive + bidirectional context
- All factorization orders during training

**Results**: Outperformed BERT on 20 tasks.

---

### 3.5 T5 (2019)

**"Exploring the Limits of Transfer Learning"**  
Raffel et al. | JMLR 2020

**Key Contributions**:
- Unified text-to-text framework
- Extensive comparison of pre-training objectives
- C4 corpus (Colossal Clean Crawled Corpus)

**Text-to-Text Example**:
```
Translation: "translate English to German: That is good" → "Das ist gut"
Classification: "sentiment: This movie is great" → "positive"
QA: "question: What is the capital of France? context: ..." → "Paris"
```

**Architecture**: Encoder-decoder Transformer (11B params)

**Results**: SOTA on SuperGLUE, SQuAD, CNN/DM.

---

### 3.6 BART (2019)

**"BART: Denoising Sequence-to-Sequence Pre-training"**  
Lewis et al. | ACL 2020

**Key Idea**: Combine BERT (denoising) + GPT (autoregressive)

**Pre-training**: Corrupt text with random transformations, then reconstruct.

**Corruptions**:
- Token masking
- Token deletion
- Sentence permutation
- Document rotation

**Architecture**: Encoder-decoder (like T5)

**Use Case**: Strong for generation tasks (summarization, translation).

---

## 4. Large Language Models (2020-2023)

### 4.1 GPT-3 (2020)

**"Language Models are Few-Shot Learners"**  
Brown et al. | NeurIPS 2020

**Scale**: 175 billion parameters

**Key Finding**: **In-context learning** emerges at scale
- Few-shot learning without gradient updates
- Prompt engineering becomes crucial

**Architecture**: Same as GPT-2, just much larger
- 96 layers
- 96 attention heads
- 12,288 hidden dimension
- 2048 context length

**Training**:
- 300B tokens
- ~$4.6M compute cost
- 3.14×10²³ FLOPs

**Results**: SOTA on many benchmarks via few-shot prompting.

**Impact**: Launched era of prompt engineering and LLM applications.

**Deep Dive**: [notebooks/03_gpt3_scaling_laws.ipynb](notebooks/03_gpt3_scaling_laws.ipynb)

---

### 4.2 Scaling Laws (2020)

**"Scaling Laws for Neural Language Models"**  
Kaplan et al. | OpenAI

**Key Findings**:

1. **Power-law relationship**:
   $$L(N) \propto N^{-\alpha}$$
   where $L$ = loss, $N$ = parameters, $\alpha \approx 0.076$

2. **Compute-optimal scaling**:
   - Model size and data should scale together
   - Don't just make model bigger

3. **Transfer**: Scaling laws hold across tasks.

**Practical Impact**: Guide for allocating compute budget.

---

### 4.3 Chinchilla (2022)

**"Training Compute-Optimal Large Language Models"**  
Hoffmann et al. | DeepMind

**Key Finding**: GPT-3 and others are **undertrained**.

**Chinchilla Scaling Law**:
$$N_{opt} \propto C^{0.50}, \quad D_{opt} \propto C^{0.50}$$

where $N$ = parameters, $D$ = tokens, $C$ = compute.

**Chinchilla-70B** vs **Gopher-280B**:
- 4× smaller
- Trained on 4× more data (1.4T tokens)
- Outperforms on most benchmarks

**Implication**: Data quality and quantity matter as much as model size.

---

### 4.4 InstructGPT (2022)

**"Training Language Models to Follow Instructions with Human Feedback"**  
Ouyang et al. | OpenAI

**Motivation**: **Alignment problem**
- Base LLMs are good at next-token prediction
- Not necessarily helpful, harmless, honest

**Three-Step Process (RLHF)**:

1. **Supervised Fine-Tuning (SFT)**:
   - Human demonstrations of desired behavior
   - Fine-tune GPT-3 on ~13k instructions

2. **Reward Model Training**:
   - Humans rank model outputs
   - Train reward model to predict human preferences

3. **RL Fine-Tuning (PPO)**:
   - Optimize policy to maximize reward

**Results**:
- InstructGPT (1.3B) preferred over GPT-3 (175B) by humans
- More helpful, less toxic

**Impact**: Blueprint for ChatGPT and GPT-4.

**Deep Dive**: [notebooks/04_rlhf_implementation.ipynb](notebooks/04_rlhf_implementation.ipynb)

---

### 4.5 PaLM (2022)

**"PaLM: Scaling Language Modeling with Pathways"**  
Chowdhery et al. | Google

**Scale**: 540 billion parameters

**Key Contributions**:
- Efficient training on TPU v4 Pods (6,144 chips)
- Breakthrough performance on reasoning tasks

**Results**:
- Few-shot: 58% on MMLU (state-of-the-art)
- Chain-of-thought reasoning: 78% on GSM8K

**Architecture Innovation**: Multi-query attention (single KV head).

---

### 4.6 LLaMA (2023)

**"LLaMA: Open and Efficient Foundation Language Models"**  
Touvron et al. | Meta

**Key Contribution**: Strong performance with smaller, more efficient models.

**Models**: 7B, 13B, 33B, 65B parameters

**Training**: 1.4T tokens (more than GPT-3)

**Results**:
- LLaMA-13B outperforms GPT-3 (175B) on most benchmarks
- LLaMA-65B competitive with Chinchilla and PaLM

**Impact**: 
- Democratized LLM research
- Enabled community fine-tuning (Alpaca, Vicuna, etc.)
- Open weights sparked ecosystem

---

### 4.7 GPT-4 (2023)

**"GPT-4 Technical Report"**  
OpenAI | 2023

**Key Features**:
- Multimodal (text + images)
- Larger context (8k / 32k tokens)
- Significantly more capable on reasoning

**Results**:
- 90th percentile on Bar exam
- 88.0% on MMLU (5-shot)
- Human-level performance on many tasks

**Safety**: Extensive RLHF and red teaming

**Limitation**: Technical details not disclosed.

---

## 5. Efficient Training & Inference (2020-2024)

### 5.1 LoRA (2021)

**"LoRA: Low-Rank Adaptation of Large Language Models"**  
Hu et al. | ICLR 2022

**Key Idea**: Freeze base model, train low-rank decomposition.

$$
W' = W + BA
$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, $r \ll d$.

**Benefits**:
- 10,000× fewer trainable parameters
- No inference latency
- Switchable adapters

**Mathematics**:
```python
# Instead of fine-tuning W (d × d):
W_finetuned = W + ΔW  # ΔW has d² parameters

# LoRA decomposes ΔW:
ΔW = B @ A  # B: d×r, A: r×d, total: 2dr parameters
# If r=8, d=4096: 2dr = 65k vs d² = 16M (250× reduction)
```

**Implementation**: [notebooks/05_lora_finetuning.ipynb](notebooks/05_lora_finetuning.ipynb)

---

### 5.2 FlashAttention (2022)

**"FlashAttention: Fast and Memory-Efficient Exact Attention"**  
Dao et al. | NeurIPS 2022

**Problem**: Standard attention is memory bandwidth bottleneck.

**Key Innovation**: Fused attention kernel (avoids HBM reads/writes).

**Algorithm**:
1. Tile attention computation
2. Keep intermediate results in SRAM
3. Reduce HBM accesses from $O(N^2)$ to $O(N)$

**Results**:
- 2-4× speedup
- Enables 64k context lengths

**Impact**: Standard in modern LLM training (GPT-4, LLaMA-2).

---

### 5.3 vLLM & PagedAttention (2023)

**"Efficient Memory Management for Large Language Model Serving"**  
Kwon et al. | SOSP 2023

**Problem**: KV cache memory fragmentation and waste.

**Key Idea**: Paged memory management (like OS virtual memory).

**Benefits**:
- 24× higher throughput than naive implementation
- Near-zero memory waste

**Impact**: State-of-the-art LLM serving framework.

---

### 5.4 QLoRA (2023)

**"QLoRA: Efficient Finetuning of Quantized LLMs"**  
Dettmers et al. | NeurIPS 2023

**Key Innovation**: Quantize base model to 4-bit + LoRA fine-tuning.

**NormalFloat (NF4)**:
- 4-bit quantization optimized for normally distributed weights
- Information-theoretically optimal

**Results**:
- Fine-tune 65B model on single 48GB GPU
- No performance degradation vs 16-bit LoRA

**Impact**: Made fine-tuning accessible on consumer hardware.

---

## 6. Alignment & Safety (2022-2024)

### 6.1 Constitutional AI (2022)

**"Constitutional AI: Harmlessness from AI Feedback"**  
Bai et al. | Anthropic

**Key Idea**: Use AI to critique and revise its own outputs.

**Process**:
1. **Supervised Phase**: Generate critiques based on constitution
2. **RL Phase**: Train preference model from AI feedback (RLAIF)

**Benefits**:
- Less reliance on human feedback
- Scalable alignment
- Transparent principles

---

### 6.2 DPO (2023)

**"Direct Preference Optimization"**  
Rafailov et al. | NeurIPS 2023

**Key Innovation**: Bypass reward model in RLHF.

**Loss Function**:
$$
\mathcal{L}_{DPO} = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
$$

**Benefits**:
- Simpler than PPO
- More stable training
- Better performance

**Impact**: Used in Zephyr, Starling models.

---

### 6.3 LIMA (2023)

**"LIMA: Less Is More for Alignment"**  
Zhou et al. | Meta

**Key Finding**: 1,000 carefully curated examples sufficient for alignment.

**Process**:
- Start with LLaMA-65B
- Fine-tune on 1,000 high-quality demonstrations
- No RLHF

**Results**: Comparable to GPT-4 on many tasks.

**Implication**: Data quality >> quantity for alignment.

---

## 7. Multimodal Models (2021-2024)

### 7.1 CLIP (2021)

**"Learning Transferable Visual Models From Natural Language Supervision"**  
Radford et al. | ICML 2021

**Key Idea**: Contrastive learning on image-text pairs.

**Training**:
- 400M image-text pairs from web
- Maximize similarity for correct pairs
- Minimize for incorrect pairs

**Loss (InfoNCE)**:
$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j) / \tau)}
$$

**Impact**: Zero-shot image classification, enables text-to-image models.

---

### 7.2 Flamingo (2022)

**"Flamingo: a Visual Language Model for Few-Shot Learning"**  
Alayrac et al. | DeepMind

**Architecture**:
- Vision encoder (CLIP)
- Cross-attention layers between vision and language
- Frozen LM

**Key Feature**: In-context learning with images.

**Results**: SOTA on vision-language tasks with few-shot prompting.

---

### 7.3 GPT-4V (2023)

**Vision-enabled GPT-4**

**Capabilities**:
- Image understanding (OCR, charts, diagrams)
- Visual reasoning
- Multimodal chain-of-thought

**Applications**: Document analysis, visual QA, image captioning.

---

### 7.4 Gemini (2024)

**"Gemini: A Family of Highly Capable Multimodal Models"**  
Google DeepMind

**Key Features**:
- Native multimodal (not bolted-on vision encoder)
- Text, images, audio, video

**Models**: Ultra (largest), Pro, Nano

**Results**: Exceeds GPT-4 on many benchmarks (MMLU: 90.0%).

---

## 8. Agents & Tool Use (2022-2024)

### 8.1 ReAct (2022)

**"ReAct: Synergizing Reasoning and Acting"**  
Yao et al. | ICLR 2023

**Key Idea**: Interleave reasoning and action.

**Example**:
```
Thought: I need to find the capital of France
Action: search("capital of France")
Observation: Paris is the capital...
Thought: Now I have the answer
Action: finish("Paris")
```

**Results**: Outperforms chain-of-thought on interactive tasks.

---

### 8.2 Toolformer (2023)

**"Toolformer: Language Models Can Teach Themselves to Use Tools"**  
Schick et al. | Meta

**Key Idea**: Self-supervised learning of tool use.

**Process**:
1. LM generates API calls in-context
2. Execute APIs and observe results
3. Keep examples where API helps perplexity
4. Fine-tune on filtered examples

**Tools**: Calculator, QA system, search, translator

---

### 8.3 Gorilla (2023)

**"Gorilla: Large Language Model Connected with Massive APIs"**  
Patil et al. | UC Berkeley

**Focus**: API calling for 1,600+ APIs (HuggingFace, PyTorch, TensorFlow).

**Dataset**: APIBench

**Results**: Outperforms GPT-4 on API accuracy.

---

## 9. Recent Breakthroughs (2024-2026)

### 9.1 Claude 3 (2024)

**Anthropic's flagship models**

**Models**: Opus, Sonnet, Haiku

**Key Features**:
- 200k context window
- Near-perfect recall
- Extended thinking mode

**Results**: Opus beats GPT-4 on many benchmarks.

---

### 9.2 Mixtral 8x7B (2024)

**"Mixtral of Experts"**  
Mistral AI

**Architecture**: Mixture of Experts (MoE)
- 8 experts, 2 active per token
- 46.7B total params, 12.9B active per token

**Results**:
- Outperforms Llama-2-70B
- 6× faster inference

**Impact**: Efficient scaling with MoE.

---

### 9.3 Groq LPU (2024)

**Hardware innovation for LLM inference**

**LPU (Language Processing Unit)**: Custom chip for sequential operations.

**Performance**: 500 tokens/second (vs 40-100 for GPUs).

**Impact**: Shows specialized hardware can dramatically improve inference.

---

### 9.4 Llama 3 (2024)

**Meta's next generation**

**Improvements**:
- Trained on 15T tokens
- Better instruction following
- Enhanced reasoning

**Models**: 8B, 70B, 405B

**Results**: Competitive with GPT-4 on many tasks.

---

### 9.5 Test-Time Compute Scaling (2024-2025)

**Key Papers**:
- "Let's Verify Step by Step" (OpenAI)
- "Quiet-STaR" (Stanford)

**Idea**: Use more compute at inference time for better reasoning.

**Methods**:
- Best-of-N sampling
- Self-consistency
- Process reward models
- Tree search

**Results**: Dramatic improvements on math/reasoning tasks.

---

### 9.6 Reasoning Models (2025-2026)

**o1 Series (OpenAI)**:
- Extended chain-of-thought during inference
- Ph.D.-level reasoning on some tasks

**Trends**:
- Inference-time scaling becomes crucial
- Verification and self-correction

---

## 10. How to Stay Current

### 10.1 Essential Resources

**ArXiv**:
- cs.CL (Computation and Language)
- cs.LG (Machine Learning)
- cs.AI (Artificial Intelligence)

**Conferences** (acceptance = quality filter):
- **NeurIPS**: Broad ML (December)
- **ICML**: Machine learning (July)
- **ICLR**: Representation learning (May)
- **ACL**: NLP (July)
- **CVPR**: Computer vision (June)
- **EMNLP**: NLP (November)

**Workshops**:
- Efficient NLP (EMNLP)
- Trustworthy ML (NeurIPS)
- Multimodal Learning (CVPR)

### 10.2 Following Researchers

**Twitter/X**: Many researchers share papers and insights

**Top ML Researchers** (follow for cutting-edge work):
- Andrej Karpathy (OpenAI, Tesla)
- Yann LeCun (Meta)
- Geoffrey Hinton (Google, Toronto)
- Yoshua Bengio (Mila)
- Ilya Sutskever (OpenAI, SSI)
- Demis Hassabis (DeepMind)
- Jeff Dean (Google)

### 10.3 Newsletters & Aggregators

- **Papers with Code**: Browse by task/sota
- **Hugging Face Daily Papers**: Curated papers
- **The Batch (deeplearning.ai)**: Weekly newsletter
- **Import AI (Jack Clark)**: Weekly roundup
- **Sebastian Raschka's Newsletter**: ML research

### 10.4 Reading Ritual

**Weekly**:
- Scan ArXiv CS.CL/CS.LG for interesting titles
- Read abstracts of 10-20 papers
- Deep dive into 1-2 most relevant

**Monthly**:
- Review accepted papers from major conferences
- Implement one influential paper

**Quarterly**:
- Survey a subfield (e.g., "efficient transformers")
- Write a blog post synthesizing findings

---

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [AlexNet Implementation](notebooks/01_alexnet.ipynb) | Reproduce ImageNet classification |
| 2 | [Transformer from Scratch](notebooks/02_transformer_from_scratch.ipynb) | Full Transformer implementation |
| 3 | [GPT-3 Scaling Laws](notebooks/03_gpt3_scaling_laws.ipynb) | Verify power-law relationships |
| 4 | [RLHF Implementation](notebooks/04_rlhf_implementation.ipynb) | Simple RLHF pipeline |
| 5 | [LoRA Fine-Tuning](notebooks/05_lora_finetuning.ipynb) | Fine-tune LLaMA with LoRA |
| 6 | [FlashAttention Analysis](notebooks/06_flashattention.ipynb) | Benchmark attention implementations |
| 7 | [CLIP Zero-Shot](notebooks/07_clip_zeroshot.ipynb) | Zero-shot image classification |
| 8 | [ReAct Agent](notebooks/08_react_agent.ipynb) | Build ReAct agent |

---

## Projects

### Mini Project: Paper Implementation Challenge
- Pick a recent paper (from last 6 months)
- Implement core algorithm
- Reproduce key result
- Write blog post explaining

### Advanced Project: Research Replication
- Choose influential paper (e.g., GPT, BERT, LoRA)
- Full replication from scratch
- Run ablation studies
- Compare to original results
- Open-source implementation

---

## Reading Lists by Subfield

### LLMs
- [x] Attention is All You Need
- [x] GPT, GPT-2, GPT-3
- [x] BERT
- [x] T5
- [x] InstructGPT
- [x] LLaMA
- [x] GPT-4
- [ ] PaLM, PaLM-2
- [ ] Gemini
- [ ] Claude 3

### Efficient Training
- [ ] Megatron-LM
- [ ] ZeRO
- [x] LoRA
- [x] QLoRA
- [x] FlashAttention
- [ ] Flash-Decoding
- [ ] Speculative Decoding

### Alignment
- [x] InstructGPT (RLHF)
- [x] Constitutional AI
- [x] DPO
- [ ] RLAIF
- [ ] RLHF scaling

### Multimodal
- [x] CLIP
- [ ] DALL-E 2
- [ ] Stable Diffusion
- [x] Flamingo
- [ ] GPT-4V

### Agents
- [x] ReAct
- [x] Toolformer
- [ ] Voyager
- [ ] AutoGPT
- [ ] WebGPT

---

## Interview Questions

### Paper Understanding
1. Explain the key innovation of the Transformer architecture.
2. What problem does LoRA solve?
3. Compare RLHF vs DPO for alignment.
4. Why is FlashAttention faster than standard attention?
5. How does CLIP enable zero-shot classification?

### Research Skills
6. How would you reproduce a paper with incomplete implementation details?
7. What makes a good ablation study?
8. Explain the difference between in-distribution and out-of-distribution generalization.
9. How do you evaluate if a model truly "understands" vs memorizes?
10. What are common pitfalls in ML research?

### Staying Current
11. How do you decide which papers to read deeply vs skim?
12. What conferences/workshops should an AI researcher follow?
13. How would you identify emerging research trends?

---

## Key Takeaways

1. **Read actively**: Question assumptions, think about alternatives
2. **Implement key papers**: Deep understanding comes from coding
3. **Track lineage**: Papers build on each other
4. **Focus on fundamentals**: Trends change, principles endure
5. **Stay current but not obsessed**: Balance breadth and depth
6. **Reproducibility matters**: Details make or break implementations
7. **Think critically**: Not all published results generalize

The field moves fast, but foundational knowledge is timeless. **Master the classics, stay curious about the new.** 🚀
