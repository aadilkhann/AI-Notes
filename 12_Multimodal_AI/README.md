# Module 12: Multimodal AI

> **Level**: Advanced  
> **Duration**: 3–4 weeks  
> **Prerequisites**: Modules 07 (Transformers), 08 (LLMs), 04 (CNNs)  
> **Goal**: Master vision-language models and multimodal architectures

---

## Table of Contents

1. [Multimodal AI Fundamentals](#1-multimodal-ai-fundamentals)
2. [Vision-Language Pretraining](#2-vision-language-pretraining)
3. [CLIP (Contrastive Language-Image Pretraining)](#3-clip-contrastive-language-image-pretraining)
4. [Vision Transformers (ViT) for Multimodal](#4-vision-transformers-vit-for-multimodal)
5. [Image Captioning](#5-image-captioning)
6. [Visual Question Answering (VQA)](#6-visual-question-answering-vqa)
7. [Large Multimodal Models (LMMs)](#7-large-multimodal-models-lmms)
8. [Text-to-Image Generation](#8-text-to-image-generation)
9. [Video Understanding](#9-video-understanding)
10. [Audio-Visual Multimodal](#10-audio-visual-multimodal)

---

## 1. Multimodal AI Fundamentals

### 1.1 What is Multimodal AI?

**Definition**: Systems that process multiple modalities (vision, language, audio).

**Modalities**:
- **Text**: Language understanding
- **Vision**: Images, videos
- **Audio**: Speech, music
- **Others**: Sensor data, code, structured data

### 1.2 Why Multimodal?

**Humans are multimodal**: We see, hear, read simultaneously.

**Benefits**:
- Richer understanding
- Cross-modal transfer
- More robust (multiple sources)

**Applications**:
- Image captioning
- Visual QA
- Text-to-image generation
- Video understanding
- Robotics

### 1.3 Key Challenges

| Challenge | Description |
|-----------|-------------|
| **Alignment** | Map text ↔ image ↔ audio |
| **Fusion** | Combine different modalities |
| **Scale** | Large paired datasets needed |
| **Architecture** | Unified model for all modalities |

### 1.4 Multimodal Fusion Strategies

**Early Fusion**: Combine raw inputs
```
[Image Features] + [Text Features] → Combined → Model
```

**Late Fusion**: Combine model outputs
```
[Image] → CNN → Score
[Text] → Transformer → Score
Weighted Sum → Final Score
```

**Cross-Modal Attention**: Attend across modalities
```
Query from Text, Key/Value from Image
```

---

## 2. Vision-Language Pretraining

### 2.1 Pretraining Objectives

**1. Image-Text Matching (ITM)**:
- Binary classification: does caption match image?

**2. Masked Language Modeling (MLM)**:
- Predict masked words conditioned on image

**3. Masked Region Modeling (MRM)**:
- Predict masked image regions

**4. Contrastive Learning**:
- Pull positive pairs together, push negatives apart

### 2.2 Datasets

| Dataset | Images | Captions |
|---------|--------|----------|
| **MS-COCO** | 330K | 1.5M |
| **Conceptual Captions** | 3.3M | 3.3M |
| **LAION-5B** | 5B | 5B |
| **Visual Genome** | 108K | 5.4M |

### 2.3 Encoder Architectures

**Dual Encoder**: Separate image + text encoders
```python
text_features = text_encoder(text)      # [B, D]
image_features = image_encoder(image)   # [B, D]
similarity = text_features @ image_features.T  # [B, B]
```

**Fusion Encoder**: Single model with cross-attention
```python
combined = fusion_encoder(image_tokens, text_tokens)
```

---

## 3. CLIP (Contrastive Language-Image Pretraining)

### 3.1 Core Idea

**Train dual encoders** to align image-text pairs in shared space.

**Objective**: Maximize similarity of matched pairs, minimize for mismatches.

### 3.2 Architecture

```
Image → Image Encoder (ViT/ResNet) → Image Embedding [D]
Text → Text Encoder (Transformer) → Text Embedding [D]
```

**Projection**: Map to shared embedding space
$$
\mathbf{z}_i = \text{Proj}_I(\text{Enc}_I(I_i)) \in \mathbb{R}^D
$$
$$
\mathbf{z}_t = \text{Proj}_T(\text{Enc}_T(T_t)) \in \mathbb{R}^D
$$

### 3.3 Contrastive Loss

**Similarity matrix**:
$$
S_{ij} = \frac{\mathbf{z}_i^I \cdot \mathbf{z}_j^T}{\|\mathbf{z}_i^I\| \|\mathbf{z}_j^T\|} \cdot \exp(\tau)
$$

Where $\tau$ is learnable temperature.

**Loss** (symmetric cross-entropy):
$$
\mathcal{L} = \frac{1}{2} \left( \mathcal{L}_{I \to T} + \mathcal{L}_{T \to I} \right)
$$

$$
\mathcal{L}_{I \to T} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(S_{ii})}{\sum_{j=1}^{N} \exp(S_{ij})}
$$

**Intuition**: Diagonal should have highest similarity.

### 3.4 Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, embed_dim=512):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        
        # Projection heads
        self.image_proj = nn.Linear(image_encoder.output_dim, embed_dim)
        self.text_proj = nn.Linear(text_encoder.output_dim, embed_dim)
        
        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def encode_image(self, images):
        features = self.image_encoder(images)
        embeddings = self.image_proj(features)
        return F.normalize(embeddings, dim=-1)
    
    def encode_text(self, texts):
        features = self.text_encoder(texts)
        embeddings = self.text_proj(features)
        return F.normalize(embeddings, dim=-1)
    
    def forward(self, images, texts):
        image_embeds = self.encode_image(images)  # [B, D]
        text_embeds = self.encode_text(texts)     # [B, D]
        
        # Cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.T  # [B, B]
        logits_per_text = logits_per_image.T
        
        return logits_per_image, logits_per_text

def clip_loss(logits_per_image, logits_per_text):
    # Ground truth: diagonal is positive
    labels = torch.arange(len(logits_per_image), device=logits_per_image.device)
    
    # Symmetric cross-entropy
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    
    return (loss_i + loss_t) / 2

# Training loop
model = CLIP(image_encoder, text_encoder)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

for images, texts in dataloader:
    logits_per_image, logits_per_text = model(images, texts)
    loss = clip_loss(logits_per_image, logits_per_text)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 3.5 Zero-Shot Classification with CLIP

**Idea**: Classify without training on specific classes.

```python
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Image
image = Image.open("dog.jpg")

# Class labels as text prompts
labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

# Encode
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

# Compute similarities
logits_per_image = outputs.logits_per_image  # [1, 3]
probs = logits_per_image.softmax(dim=1)

# Print results
for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob:.4f}")
# Output:
# a photo of a cat: 0.0123
# a photo of a dog: 0.9654
# a photo of a car: 0.0223
```

### 3.6 Applications of CLIP

- **Zero-shot classification**
- **Image retrieval** (text → similar images)
- **Text retrieval** (image → similar captions)
- **Backbone for larger models** (Flamingo, DALL-E 2)

---

## 4. Vision Transformers (ViT) for Multimodal

### 4.1 Patch Embedding

**Split image into patches**:
$$
\text{Image } (H, W, C) \to \text{Patches } (N, P^2 \cdot C)
$$

Where $N = \frac{HW}{P^2}$ (e.g., $224 \times 224$ with $P=16$ → $196$ patches).

**Linear projection**:
$$
\mathbf{x}_i = \text{Linear}(\text{Patch}_i) \in \mathbb{R}^D
$$

### 4.2 Cross-Modal Attention

**Query from text, Key/Value from image**:
```python
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
    
    def forward(self, text_features, image_features):
        B, N_t, D = text_features.shape
        B, N_i, D = image_features.shape
        
        # Q from text, K/V from image
        Q = self.q_proj(text_features).reshape(B, N_t, self.num_heads, -1).transpose(1, 2)
        K = self.k_proj(image_features).reshape(B, N_i, self.num_heads, -1).transpose(1, 2)
        V = self.v_proj(image_features).reshape(B, N_i, self.num_heads, -1).transpose(1, 2)
        
        # Attention
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # [B, H, N_t, N_i]
        attn = attn.softmax(dim=-1)
        
        out = attn @ V  # [B, H, N_t, D/H]
        out = out.transpose(1, 2).reshape(B, N_t, D)
        
        return self.out_proj(out)
```

---

## 5. Image Captioning

### 5.1 Encoder-Decoder Architecture

```
Image → CNN/ViT → Visual Features → Transformer Decoder → Caption
```

**Encoder**: Extract visual features  
**Decoder**: Generate text conditioned on image

### 5.2 Implementation

```python
class ImageCaptioningModel(nn.Module):
    def __init__(self, image_encoder, vocab_size, hidden_dim=512, num_layers=6):
        super().__init__()
        self.image_encoder = image_encoder
        
        # Project image features to text space
        self.proj = nn.Linear(image_encoder.output_dim, hidden_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(512, hidden_dim)
    
    def forward(self, images, captions):
        # Encode image
        img_features = self.image_encoder(images)  # [B, N, D]
        memory = self.proj(img_features)           # [B, N, H]
        memory = memory.transpose(0, 1)            # [N, B, H]
        
        # Embed captions
        B, T = captions.shape
        positions = torch.arange(T, device=captions.device).unsqueeze(0).expand(B, -1)
        tgt = self.token_embed(captions) + self.pos_embed(positions)  # [B, T, H]
        tgt = tgt.transpose(0, 1)  # [T, B, H]
        
        # Causal mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(captions.device)
        
        # Decode
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)  # [T, B, H]
        logits = self.output_proj(output.transpose(0, 1))      # [B, T, V]
        
        return logits
    
    @torch.no_grad()
    def generate(self, image, max_length=50, temperature=1.0):
        img_features = self.image_encoder(image.unsqueeze(0))
        memory = self.proj(img_features).transpose(0, 1)  # [N, 1, H]
        
        # Start with <BOS>
        caption = [self.tokenizer.bos_token_id]
        
        for _ in range(max_length):
            tgt = self.token_embed(torch.tensor([caption])) + \
                  self.pos_embed(torch.arange(len(caption)).unsqueeze(0))
            tgt = tgt.transpose(0, 1)
            
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(caption))
            
            output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            logits = self.output_proj(output[-1]) / temperature
            
            next_token = logits.argmax(dim=-1).item()
            caption.append(next_token)
            
            if next_token == self.tokenizer.eos_token_id:
                break
        
        return caption
```

### 5.3 Training

```python
from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTFeatureExtractor

# Load pretrained encoder-decoder
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    "google/vit-base-patch16-224-in21k",  # Encoder
    "gpt2"                                 # Decoder
)

# Set decoder to cross-attention mode
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# Training
for batch in dataloader:
    images, captions = batch
    outputs = model(pixel_values=images, labels=captions)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

---

## 6. Visual Question Answering (VQA)

### 6.1 Task Definition

**Input**: Image + Question  
**Output**: Answer

**Example**:
- Image: Dog on beach
- Question: "What color is the dog?"
- Answer: "Brown"

### 6.2 Architecture

```python
class VQAModel(nn.Module):
    def __init__(self, image_encoder, text_encoder, num_answers):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        
        # Cross-modal fusion
        self.cross_attention = CrossAttention(dim=512)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_answers)
        )
    
    def forward(self, images, questions):
        # Encode
        img_feats = self.image_encoder(images)      # [B, N_i, D]
        text_feats = self.text_encoder(questions)   # [B, N_t, D]
        
        # Cross-attention (text queries image)
        fused = self.cross_attention(text_feats, img_feats)  # [B, N_t, D]
        
        # Pool and classify
        pooled = fused.mean(dim=1)  # [B, D]
        logits = self.classifier(pooled)  # [B, num_answers]
        
        return logits
```

---

## 7. Large Multimodal Models (LMMs)

### 7.1 Flamingo

**Architecture**: Interleave vision and language

```
[Image] → Vision Encoder → Perceiver Resampler
   ↓
[Text] → LLM with gated cross-attention → Output
```

**Key innovation**: Gated cross-attention layers
$$
\text{Output} = \text{FFN}(\text{SelfAttn}(x) + \tanh(\alpha) \cdot \text{CrossAttn}(x, \text{vision}))
$$

Where $\alpha$ is learned gate (starts at 0).

### 7.2 LLaVA (Large Language-and-Vision Assistant)

**Simple design**: Vision encoder + projection + LLM

```
Image → CLIP ViT → Linear Projection → LLaMA
         ↓
    [visual tokens]
```

**Training**:
1. **Stage 1**: Train projection layer (freeze encoder + LLM)
2. **Stage 2**: Fine-tune full model on instruction data

```python
class LLaVA(nn.Module):
    def __init__(self, vision_encoder, llm, projection_dim=4096):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.llm = llm
        
        # Project vision to LLM space
        self.mm_projector = nn.Linear(vision_encoder.hidden_size, projection_dim)
    
    def forward(self, images, input_ids):
        # Encode image
        image_features = self.vision_encoder(images).last_hidden_state  # [B, N, D_v]
        image_tokens = self.mm_projector(image_features)  # [B, N, D_llm]
        
        # Embed text
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # [B, T, D_llm]
        
        # Concatenate [image_tokens, text_embeds]
        inputs_embeds = torch.cat([image_tokens, text_embeds], dim=1)
        
        # Forward through LLM
        outputs = self.llm(inputs_embeds=inputs_embeds)
        return outputs
```

### 7.3 GPT-4V (GPT-4 Vision)

**Capabilities** (from examples):
- Visual reasoning
- OCR and document understanding
- Chart/diagram interpretation
- Spatial reasoning
- Code generation from UI mockups

**Architecture**: Not public, likely similar to Flamingo/LLaVA approach.

### 7.4 Gemini

**Multimodal from scratch**: Trained on interleaved text, image, audio, video.

**Key features**:
- Native multimodal (not bolted on)
- Long context (up to 1M tokens)
- Video understanding

---

## 8. Text-to-Image Generation

### 8.1 DALL-E 2

**Architecture**:
```
Text → CLIP Text Encoder → Prior → CLIP Image Embedding → Decoder → Image
```

**Prior**: Maps text embedding to image embedding (diffusion model)

**Decoder**: Diffusion model conditioned on CLIP embedding

### 8.2 Stable Diffusion

**Latent Diffusion Model**:
```
Text → CLIP → Conditioning
Image → VAE Encoder → Latent → Diffusion → Latent → VAE Decoder → Image
```

**Why latent space?**
- **Efficiency**: $512 \times 512$ → $64 \times 64$ latent (64× fewer pixels)
- **Quality**: Smooth latent space

**UNet with Cross-Attention**:
```python
class CrossAttnUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = nn.Sequential(...)
        self.down_blocks = nn.ModuleList([...])
        self.cross_attn_blocks = nn.ModuleList([...])
        self.up_blocks = nn.ModuleList([...])
    
    def forward(self, latent, timestep, text_embed):
        # Time embedding
        t_emb = self.time_embed(timestep)
        
        # Downsampling with cross-attention
        for down, cross_attn in zip(self.down_blocks, self.cross_attn_blocks):
            latent = down(latent, t_emb)
            latent = cross_attn(latent, text_embed)  # Condition on text
        
        # Upsampling
        for up in self.up_blocks:
            latent = up(latent, t_emb)
        
        return latent
```

---

## 9. Video Understanding

### 9.1 Challenges

- **Temporal dimension**: Not just images
- **Long sequences**: Videos are 30+ fps
- **Memory**: Cannot load all frames

### 9.2 Frame Sampling Strategies

| Strategy | Description |
|----------|-------------|
| **Uniform** | Sample every N frames |
| **Random** | Random frames during training |
| **Keyframes** | Detect scene changes |
| **Hierarchical** | Coarse-to-fine sampling |

### 9.3 Video Transformer

**Patch + Time Embedding**:
```python
class VideoViT(nn.Module):
    def __init__(self, num_frames=8, patch_size=16):
        super().__init__()
        self.num_frames = num_frames
        
        # Patch embedding (same as ViT)
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Temporal embedding
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, 1, embed_dim))
        
        # Transformer
        self.transformer = nn.TransformerEncoder(...)
    
    def forward(self, video):
        # video: [B, T, C, H, W]
        B, T, C, H, W = video.shape
        
        # Flatten along batch and time
        video = video.reshape(B * T, C, H, W)
        
        # Patch embedding
        patches = self.patch_embed(video)  # [B*T, D, H', W']
        patches = patches.flatten(2).transpose(1, 2)  # [B*T, N, D]
        
        # Reshape to include time
        patches = patches.reshape(B, T, -1, patches.size(-1))  # [B, T, N, D]
        
        # Add temporal embedding
        patches = patches + self.time_embed
        
        # Flatten for transformer
        patches = patches.reshape(B, T * patches.size(2), -1)  # [B, T*N, D]
        
        # Transformer
        output = self.transformer(patches)
        return output
```

---

## 10. Audio-Visual Multimodal

### 10.1 Speech + Vision

**Applications**:
- Lip reading
- Audio-visual speech recognition
- Speaker diarization with face detection

### 10.2 Audio-Visual Correspondence

**Task**: Do audio and video match?

**Architecture**:
```python
class AVCorrespondence(nn.Module):
    def __init__(self, audio_encoder, video_encoder):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder
        
        self.audio_proj = nn.Linear(audio_dim, 512)
        self.video_proj = nn.Linear(video_dim, 512)
    
    def forward(self, audio, video):
        audio_feat = F.normalize(self.audio_proj(self.audio_encoder(audio)))
        video_feat = F.normalize(self.video_proj(self.video_encoder(video)))
        
        # Cosine similarity
        similarity = (audio_feat * video_feat).sum(dim=-1)
        return similarity
```

---

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [CLIP from Scratch](notebooks/01_clip.ipynb) | Implement and train CLIP |
| 2 | [Zero-Shot Classification](notebooks/02_zero_shot.ipynb) | Use CLIP for classification |
| 3 | [Image Captioning](notebooks/03_image_captioning.ipynb) | Build captioning model |
| 4 | [VQA](notebooks/04_vqa.ipynb) | Visual question answering |
| 5 | [LLaVA Fine-Tuning](notebooks/05_llava.ipynb) | Fine-tune multimodal LLM |

---

## Projects

### Mini Project: Visual Search Engine
- Use CLIP to build image search
- Text queries → Find similar images
- Image queries → Find similar images
- Deploy with Streamlit

### Advanced Project: Multimodal Assistant
- Combine LLaVA + Stable Diffusion
- Accept text + image inputs
- Generate text or image outputs
- Chat interface with memory

---

## Interview Questions

1. Explain CLIP and how contrastive learning works.
2. What's the difference between dual encoder and fusion encoder?
3. Walk through the image captioning architecture.
4. How does cross-modal attention differ from self-attention?
5. Explain zero-shot classification with CLIP.
6. What are the challenges in video understanding?
7. How does Stable Diffusion condition on text?
8. Compare Flamingo and LLaVA architectures.
9. Why use latent diffusion instead of pixel-space diffusion?
10. How would you build a VQA system from scratch?
