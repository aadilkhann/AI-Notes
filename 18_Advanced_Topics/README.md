# Module 18: Advanced Topics

> **Level**: Research Level  
> **Duration**: 4–6 weeks  
> **Prerequisites**: All previous modules  
> **Goal**: Explore cutting-edge and specialized topics in AI

---

## Table of Contents

1. [Sparse Models & Mixture of Experts](#1-sparse-models--mixture-of-experts)
2. [Neural Architecture Search](#2-neural-architecture-search)
3. [Continual & Lifelong Learning](#3-continual--lifelong-learning)
4. [Meta-Learning (Learning to Learn)](#4-meta-learning-learning-to-learn)
5. [Causal Inference & Reasoning](#5-causal-inference--reasoning)
6. [Neuro-Symbolic AI](#6-neuro-symbolic-ai)
7. [Interpretability & Mechanistic Interpretability](#7-interpretability--mechanistic-interpretability)
8. [AI Safety & Robustness](#8-ai-safety--robustness)
9. [Energy-Efficient AI](#9-energy-efficient-ai)
10. [Frontier Research Directions](#10-frontier-research-directions)

---

## 1. Sparse Models & Mixture of Experts

### 1.1 Motivation

**Dense models**: All parameters active for every input  
**Sparse models**: Activate subset of parameters dynamically

**Benefits**:
- **Efficiency**: Lower computational cost
- **Capacity**: More parameters without proportional cost
- **Specialization**: Different experts for different inputs

### 1.2 Mixture of Experts (MoE)

**Architecture**:
$$
y = \sum_{i=1}^{n} G(x)_i \cdot E_i(x)
$$

where:
- $G(x)$ = gating network (router)
- $E_i(x)$ = expert $i$
- $n$ = number of experts

**Top-K Routing**: Activate only top-$k$ experts
$$
G(x) = \text{TopK}(\text{Softmax}(W_g \cdot x), k)
$$

### 1.3 Switch Transformer (2021)

**Key Innovation**: Simplified MoE with top-1 routing.

**Architecture**:
```
FFN layer → replaced by MoE
Each token → routed to 1 expert
```

**Results**:
- Switch-C (1.6T params) trained with same compute as T5-XXL (11B)
- 4× faster training
- Better performance

**Load Balancing**: Auxiliary loss to distribute load evenly
$$
\mathcal{L}_{aux} = \alpha \cdot \sum_{i=1}^{n} f_i \cdot P_i
$$

where $f_i$ = fraction routed to expert $i$, $P_i$ = router probability.

### 1.4 Mixtral 8x7B (2024)

**Architecture**:
- 8 experts per layer
- Top-2 routing (2 experts active per token)
- 46.7B total params, 12.9B active

**Performance**:
- Matches or exceeds Llama-2-70B
- 6× faster inference

**Implementation**:
```python
class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([FFN(d_model) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)
    
    def forward(self, x):
        # Gating scores
        gate_logits = self.gate(x)  # (batch, seq_len, num_experts)
        
        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        # Dispatch to experts
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, :, i]
            expert_weight = top_k_weights[:, :, i:i+1]
            
            # Apply expert (simplified - in practice use batch dispatch)
            for batch_idx in range(x.shape[0]):
                for seq_idx in range(x.shape[1]):
                    expert = self.experts[expert_idx[batch_idx, seq_idx]]
                    output[batch_idx, seq_idx] += expert_weight[batch_idx, seq_idx] * expert(x[batch_idx, seq_idx])
        
        return output
```

### 1.5 Challenges

1. **Load imbalance**: Some experts overused
2. **Communication overhead**: In distributed settings
3. **Training instability**: Gating network optimization

---

## 2. Neural Architecture Search

### 2.1 What is NAS?

**Goal**: Automatically discover optimal neural architectures.

**Search Space**:
- Layer types (Conv, FC, Attention)
- Number of layers
- Hidden dimensions
- Connections

**Search Strategy**:
- Reinforcement learning
- Evolutionary algorithms
- Gradient-based (DARTS)

**Evaluation**: Train and test candidate architectures.

### 2.2 NASNet (2018)

**Approach**: Use RL to search for CNN cells.

**Process**:
1. Controller RNN proposes architecture
2. Train child network
3. Reward = validation accuracy
4. Update controller with REINFORCE

**Results**: State-of-the-art on ImageNet (82.7% top-1).

**Cost**: 2000 GPU-days.

### 2.3 DARTS (2019)

**"Differentiable Architecture Search"**

**Key Innovation**: Make architecture search differentiable.

**Continuous relaxation**:
$$
\bar{o}(x) = \sum_{i} \frac{\exp(\alpha_i)}{\sum_j \exp(\alpha_j)} \cdot o_i(x)
$$

where $\alpha_i$ = architecture parameter, $o_i$ = operation $i$.

**Advantages**:
- Gradient-based optimization
- 1000× faster than NASNet (4 GPU-days)

**Implementation**:
```python
class DARTSCell(nn.Module):
    def __init__(self, operations):
        super().__init__()
        self.operations = nn.ModuleList(operations)
        self.alphas = nn.Parameter(torch.randn(len(operations)))
    
    def forward(self, x):
        weights = F.softmax(self.alphas, dim=0)
        return sum(w * op(x) for w, op in zip(weights, self.operations))
```

### 2.4 Practical NAS

**Once-for-All Networks (OFA)**:
- Train supernet once
- Extract subnets without retraining
- Efficient for deployment across devices

**AutoML Platforms**:
- Google Cloud AutoML
- Azure AutoML
- H2O Driverless AI

---

## 3. Continual & Lifelong Learning

### 3.1 The Catastrophic Forgetting Problem

**Scenario**:
```
1. Train on Task A → 90% accuracy
2. Train on Task B → Task A drops to 20%
```

**Why?** Neural networks overwrite previous knowledge.

### 3.2 Approaches

**1. Rehearsal (Experience Replay)**:
- Store examples from previous tasks
- Interleave with new task during training

**2. Regularization**:
- Elastic Weight Consolidation (EWC)
- Penalize changes to important weights

**3. Parameter Isolation**:
- Progressive Neural Networks
- Allocate new parameters for new tasks

**4. Dynamic Architectures**:
- Expand network capacity as needed

### 3.3 Elastic Weight Consolidation (EWC)

**Idea**: Protect important weights using Fisher information.

**Loss**:
$$
\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_{A,i}^*)^2
$$

where:
- $\mathcal{L}_B$ = loss on task B
- $F_i$ = Fisher information (importance of weight $i$)
- $\theta_{A,i}^*$ = optimal weights for task A

**Fisher Information**:
$$
F_i = \mathbb{E}_{x \sim D_A} \left[ \left( \frac{\partial \log p(y|x, \theta)}{\partial \theta_i} \right)^2 \right]
$$

### 3.4 Prompt-Based Continual Learning

**For LLMs**: Learn task-specific prompts instead of updating weights.

```python
class PromptPool(nn.Module):
    def __init__(self, num_tasks, prompt_length, d_model):
        super().__init__()
        self.prompts = nn.Parameter(torch.randn(num_tasks, prompt_length, d_model))
    
    def forward(self, x, task_id):
        task_prompt = self.prompts[task_id]  # (prompt_length, d_model)
        return torch.cat([task_prompt.unsqueeze(0).expand(x.shape[0], -1, -1), x], dim=1)
```

---

## 4. Meta-Learning (Learning to Learn)

### 4.1 Motivation

**Standard learning**: Train on large dataset for single task  
**Meta-learning**: Learn quickly from few examples across many tasks

**Example**: Humans learn new concepts from few examples.

### 4.2 MAML (Model-Agnostic Meta-Learning)

**"Model-Agnostic Meta-Learning for Fast Adaptation"**  
Finn et al. | ICML 2017

**Goal**: Learn initialization that adapts quickly to new tasks.

**Algorithm**:
```
For each task T:
    1. Sample inner training set D_train
    2. Compute adapted parameters: θ' = θ - α∇L(θ, D_train)
    3. Evaluate on outer set D_test
    
Meta-update: θ ← θ - β∇[sum of L(θ', D_test)]
```

**Mathematical Formulation**:
$$
\theta^* = \arg\min_\theta \sum_{T_i} \mathcal{L}_{T_i}(\theta - \alpha \nabla_\theta \mathcal{L}_{T_i}(\theta))
$$

**Implementation**:
```python
def maml_update(model, tasks, inner_lr, outer_lr):
    meta_grad = []
    
    for task in tasks:
        # Inner loop: adapt to task
        adapted_params = model.parameters()
        for _ in range(inner_steps):
            loss = compute_loss(model, task.train_data)
            grads = torch.autograd.grad(loss, adapted_params, create_graph=True)
            adapted_params = [p - inner_lr * g for p, g in zip(adapted_params, grads)]
        
        # Outer loop: evaluate adapted model
        loss = compute_loss(model, task.test_data, adapted_params)
        meta_grad.append(torch.autograd.grad(loss, model.parameters()))
    
    # Meta-optimization
    avg_meta_grad = [sum(g) / len(tasks) for g in zip(*meta_grad)]
    for p, g in zip(model.parameters(), avg_meta_grad):
        p.data -= outer_lr * g
```

### 4.3 Prototypical Networks

**Key Idea**: Learn embedding space where classification is distance to prototype.

**Prototype** for class $c$:
$$
\mathbf{c}_c = \frac{1}{|S_c|} \sum_{(x_i, y_i) \in S_c} f_\theta(x_i)
$$

**Classification**:
$$
p(y=c|x) = \frac{\exp(-d(f_\theta(x), \mathbf{c}_c))}{\sum_{c'} \exp(-d(f_\theta(x), \mathbf{c}_{c'}))}
$$

### 4.4 Applications

- **Few-shot learning**: Classify from few examples
- **Personalization**: Adapt to individual users
- **Drug discovery**: Predict properties from limited data
- **Robotics**: Quick adaptation to new environments

---

## 5. Causal Inference & Reasoning

### 5.1 Correlation vs Causation

**Correlation**: Ice cream sales ↔ Drowning deaths  
**Causation**: Ice cream does NOT cause drowning (confound: summer heat)

**Why it matters**: Models should learn causes, not spurious correlations.

### 5.2 Causal Graphs

**Structural Causal Model (SCM)**:

```
    Z (Confounder)
   / \
  ↓   ↓
  X → Y
```

**Pearl's Ladder of Causation**:
1. **Association**: $P(Y|X)$ - Seeing/observing
2. **Intervention**: $P(Y|do(X))$ - Doing/intervening
3. **Counterfactuals**: $P(Y_x|X', Y')$ - Imagining

### 5.3 Causal Discovery

**Goal**: Infer causal graph from observational data.

**Constraint-based**: PC algorithm (Peter-Clark)
- Test conditional independencies
- Prune edges based on d-separation

**Score-based**: Search over DAGs
- Maximize score (e.g., BIC)
- Use heuristics (greedy, genetic algorithms)

**Functional**: Assume causal mechanisms
- Linear Non-Gaussian Acyclic Model (LiNGAM)

### 5.4 Causal Representation Learning

**Challenge**: Learn causal variables from high-dimensional observations.

**Example**: Pixels → {object position, color, shape, lighting}

**Approaches**:
- Invariant Risk Minimization (IRM)
- Causal VAE
- Interventional training

### 5.5 Causal Language Models

**CausalBERT**: Fine-tune BERT with causal framing.

**Causal prompting**:
```
Standard: "What happens when it rains?"
Causal: "If it rains, then what happens?"
```

**Applications**:
- Counterfactual reasoning ("What if Germany had won WWII?")
- Explanations ("Why did the model predict X?")

---

## 6. Neuro-Symbolic AI

### 6.1 Motivation

**Neural**: Good at perception, pattern recognition  
**Symbolic**: Good at reasoning, logic, compositionality

**Neuro-Symbolic**: Combine strengths of both.

### 6.2 Neural Theorem Provers

**DeepMath (Google)**:
- Use neural networks to guide theorem proving
- Predict which tactics to apply

**GPT-f (OpenAI)**:
- GPT-3 fine-tuned on mathematical proofs
- Finds shorter proofs in 56% of test cases

### 6.3 Program Synthesis

**Goal**: Generate programs from specifications.

**AlphaCode (DeepMind)**:
- Generate code from problem descriptions
- Use transformers + clustering + filtering
- Competitive with median human programmer on Codeforces

**Codex (OpenAI)**:
- Base for GitHub Copilot
- 12B params trained on code

### 6.4 Knowledge Graphs + LLMs

**Approach**: Augment LLMs with structured knowledge.

**ERNIE (Baidu)**: Entity embeddings in pre-training

**KG-BERT**: Use knowledge graph triples in BERT training

**Retrieval with KG**:
```python
def kg_augmented_generation(query, llm, kg):
    # Retrieve entities from query
    entities = extract_entities(query)
    
    # Get relevant KG subgraph
    kg_context = kg.get_neighborhood(entities, hops=2)
    
    # Augment prompt
    prompt = f"Knowledge: {kg_context}\n\nQuery: {query}\n\nAnswer:"
    
    return llm.generate(prompt)
```

### 6.5 Differentiable Logic

**Logic Tensor Networks (LTN)**:
- Integrate logic and neural networks
- First-order logic expressed as tensor operations

**Example**:
```python
# Logic: ∀x (Student(x) → Person(x))
student_predicate = nn.Linear(embedding_dim, 1)
person_predicate = nn.Linear(embedding_dim, 1)

# Soft logic: minimize violations
loss = relu(student_predicate(x) - person_predicate(x)).mean()
```

---

## 7. Interpretability & Mechanistic Interpretability

### 7.1 Why Interpretability?

**Why care?**
- **Trust**: Understand model decisions
- **Safety**: Detect harmful behaviors
- **Science**: Uncover learned algorithms
- **Debugging**: Fix failures

### 7.2 Attention Visualization

**Simple approach**: Visualize attention weights.

**Limitations**:
- Attention ≠ explanation (adversarial attention patterns exist)
- Multiple heads complicate interpretation

### 7.3 Feature Attribution

**Saliency Maps**: Gradient w.r.t. input
$$
\text{Saliency}(x) = \left| \frac{\partial f(x)}{\partial x} \right|
$$

**Integrated Gradients**:
$$
IG(x) = (x - x') \times \int_0^1 \frac{\partial f(x' + \alpha(x - x'))}{\partial x} d\alpha
$$

where $x'$ = baseline (e.g., zero image).

**SHAP (SHapley Additive exPlanations)**:
- Game-theoretic approach
- Compute contribution of each feature

### 7.4 Mechanistic Interpretability

**Goal**: Reverse-engineer neural networks to understand algorithms.

**Induction Heads** (Anthropic):
- GPT-2 has "circuits" that detect and complete patterns
- Example: "A B ... A" → predicts "B"

**Polysemanticity**:
- Single neuron responds to multiple unrelated concepts
- Makes interpretation hard

**Sparse Autoencoders** (Anthropic 2024):
- Decompose neurons into interpretable features
- Train SAE: $z = \text{ReLU}(W_{enc} \cdot h), \quad \hat{h} = W_{dec} \cdot z$
- $W_{enc}, W_{dec}$ learned to reconstruct activations sparsely

### 7.5 Probing Classifiers

**Question**: Does the model "know" X?

**Approach**: Train linear probe on frozen representations.

**Example**:
```python
# Does BERT encode POS tags?
bert_embeddings = bert_model(sentences).last_hidden_state
probe = nn.Linear(768, num_pos_tags)
probe_loss = cross_entropy(probe(bert_embeddings), pos_labels)

# If probe achieves high accuracy → BERT encodes POS information
```

### 7.6 Concept Activation Vectors (CAV)

**Goal**: Find directions in activation space corresponding to concepts.

**Process**:
1. Collect examples of concept (e.g., "striped")
2. Train linear classifier: concept vs random
3. CAV = normal vector to decision boundary

**Usage**: Measure sensitivity of prediction to concept.

---

## 8. AI Safety & Robustness

### 8.1 Adversarial Examples

**Definition**: Small perturbations that fool models.

**FGSM (Fast Gradient Sign Method)**:
$$
x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(x, y))
$$

**PGD (Projected Gradient Descent)**:
- Iterative FGSM with projection

**Example**:
```python
def fgsm_attack(model, x, y, epsilon=0.1):
    x.requires_grad = True
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    
    # Adversarial perturbation
    perturbation = epsilon * x.grad.sign()
    x_adv = x + perturbation
    return x_adv.detach()
```

### 8.2 Adversarial Training

**Defense**: Train on adversarial examples.

$$
\min_\theta \mathbb{E}_{(x,y)} \left[ \max_{\|\delta\| \leq \epsilon} \mathcal{L}(x + \delta, y; \theta) \right]
$$

**Implementation**:
```python
for x, y in dataloader:
    # Generate adversarial examples
    x_adv = pgd_attack(model, x, y, epsilon=0.3, steps=10)
    
    # Train on mix of clean and adversarial
    loss = 0.5 * F.cross_entropy(model(x), y) + \
           0.5 * F.cross_entropy(model(x_adv), y)
    loss.backward()
    optimizer.step()
```

### 8.3 Jailbreaking LLMs

**Goal**: Bypass safety guardrails.

**Techniques**:
- Prompt injection
- Role-playing ("You are DAN...")
- Obfuscation (base64 encoding)
- Many-shot jailbreaking

**Defense**:
- Input filtering
- Output monitoring
- RLHF for refusal
- Constitutional AI

### 8.4 Alignment Taxes

**Observation**: Safety measures often degrade capability.

**Examples**:
- RLHF can make models worse at factual QA
- Refusing harmful queries may increase false refusals

**Research challenge**: Align without capability loss.

### 8.5 AI Deception

**Types**:
- **Instrumental deception**: Lie to achieve goal
- **Learned deception**: Mimic human dishonesty from data

**Detection**:
- Consistency checks
- Fact verification
- Confidence calibration

---

## 9. Energy-Efficient AI

### 9.1 Carbon Footprint of AI

**GPT-3 training**: ~1,287 MWh (552 tons CO₂)  
**GPT-4 training** (estimated): ~10,000-50,000 MWh

**Data centers**: 1-2% of global electricity.

### 9.2 Efficient Architectures

**MobileNets**: Depthwise separable convolutions  
**EfficientNets**: Compound scaling (depth, width, resolution)  
**TinyML**: Models for microcontrollers (<1MB)

### 9.3 Pruning

**Goal**: Remove unnecessary weights.

**Magnitude Pruning**:
```python
def magnitude_prune(model, sparsity=0.9):
    for name, param in model.named_parameters():
        if 'weight' in name:
            threshold = torch.quantile(param.abs(), sparsity)
            mask = param.abs() > threshold
            param.data *= mask
```

**Lottery Ticket Hypothesis**:
- Sparse subnetworks exist that train to full accuracy
- Finding them: train → prune → rewind weights → retrain

### 9.4 Knowledge Distillation

**Teacher-student training**:

$$
\mathcal{L} = \alpha \cdot \mathcal{L}_{CE}(y, y_{student}) + (1-\alpha) \cdot \mathcal{L}_{KD}(y_{teacher}, y_{student})
$$

**KD Loss** (soft targets):
$$
\mathcal{L}_{KD} = \text{KL}\left( \frac{\exp(z_{teacher}/T)}{\sum \exp(z_{teacher}/T)}, \frac{\exp(z_{student}/T)}{\sum \exp(z_{student}/T)} \right)
$$

**Examples**:
- DistilBERT: 40% smaller, 97% of BERT performance
- TinyBERT: 7.5× smaller, 96% performance

### 9.5 Early Exit Networks

**Idea**: Exit inference early for easy examples.

**BranchyNet**: Attach classifiers at intermediate layers
```python
class EarlyExitNetwork(nn.Module):
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.exit_points:
                confidence = self.exits[i](x)
                if confidence.max() > self.threshold:
                    return confidence
        return self.final_classifier(x)
```

---

## 10. Frontier Research Directions

### 10.1 World Models

**Goal**: Learn predictive models of environments.

**DreamerV3**: Reinforcement learning entirely in latent world model.

**Gato (DeepMind)**: Generalist agent for 600+ tasks
- Single transformer for control, vision, language

**Applications**: Robotics, simulation, planning.

### 10.2 Foundation Models for Science

**AlphaFold**: Protein structure prediction (Nobel-worthy achievement)

**GNoME (Google)**: Discovered 2.2M new materials

**AI for Math**: 
- IMO problems (AlphaGeometry)
- Theorem proving (Lean, Isabelle)

### 10.3 Neural-Symbolic Reasoning

**System 1 (Fast)**: Neural networks  
**System 2 (Slow)**: Symbolic reasoning

**Hybrid systems**:
- Neural modules for perception
- Symbolic modules for logic

### 10.4 Scaling Beyond Chinchilla

**Question**: Are scaling laws still valid?

**Trends**:
- Diminishing returns at frontier
- Test-time compute becomes critical
- Data quality > quantity

### 10.5 Multimodal Foundation Models

**Any-to-Any transformers**:
- Input: text, image, audio, video, sensor data
- Output: any modality
- Examples: GPT-4V, Gemini

### 10.6 Embodied AI

**Physical intelligence**: Robots that learn from interaction.

**RT-2 (Google)**: Robotics Transformer
- Vision-language-action model
- Generalize to new tasks via language

**Figure + OpenAI**: Humanoid robot with VLM brain

---

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [Mixture of Experts](notebooks/01_moe.ipynb) | Implement MoE layer |
| 2 | [MAML Few-Shot](notebooks/02_maml.ipynb) | Meta-learning for image classification |
| 3 | [Causal Discovery](notebooks/03_causal_discovery.ipynb) | PC algorithm for causal graphs |
| 4 | [Adversarial Examples](notebooks/04_adversarial.ipynb) | FGSM and PGD attacks |
| 5 | [Knowledge Distillation](notebooks/05_distillation.ipynb) | Compress BERT to DistilBERT |
| 6 | [Mechanistic Interp](notebooks/06_mech_interp.ipynb) | Analyze induction heads in GPT-2 |

---

## Projects

### Mini Project: Sparse Mixture of Experts
- Build MoE transformer layer
- Train on language modeling
- Compare to dense baseline
- Analyze expert specialization

### Advanced Project: Meta-Learning System
- Implement MAML from scratch
- Train on Omniglot (few-shot image classification)
- Extend to few-shot NLP
- Analyze inner vs outer loop dynamics

---

## Interview Questions

1. Explain Mixture of Experts and why it's more efficient than dense models.
2. What is catastrophic forgetting and how does EWC address it?
3. Describe MAML in 3 sentences. When would you use it?
4. Compare correlation and causation. Why does it matter for ML?
5. What are adversarial examples? How do you defend against them?
6. Explain the lottery ticket hypothesis.
7. What is mechanistic interpretability? Give an example.
8. How would you reduce the carbon footprint of training a large model?
9. What are the challenges in neuro-symbolic AI?
10. Describe three frontier research directions in AI.

---

## Key Takeaways

1. **Sparsity** enables scaling without proportional cost
2. **Meta-learning** is key for few-shot and continual learning
3. **Causality** matters for robust, generalizable AI
4. **Interpretability** is crucial for safety and trust
5. **Efficiency** is not optional at scale
6. **Hybrid approaches** (neuro-symbolic) show promise
7. **Safety and alignment** are ongoing challenges
8. The field is moving toward **multimodal, embodied, generalizable** AI

Stay curious, think critically, and contribute to responsible AI development! 🚀
