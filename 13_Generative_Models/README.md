# Module 13: Generative Models

> **Level**: Advanced  
> **Duration**: 4–5 weeks  
> **Prerequisites**: Modules 03 (Deep Learning), 04 (CNNs), 14 (Reinforcement Learning for advanced topics)  
> **Goal**: Master VAEs, GANs, Diffusion Models, and modern generative architectures

---

## Table of Contents

1. [Generative Models Fundamentals](#1-generative-models-fundamentals)
2. [Autoencoders](#2-autoencoders)
3. [Variational Autoencoders (VAEs)](#3-variational-autoencoders-vaes)
4. [Generative Adversarial Networks (GANs)](#4-generative-adversarial-networks-gans)
5. [Advanced GAN Architectures](#5-advanced-gan-architectures)
6. [Diffusion Models](#6-diffusion-models)
7. [Denoising Diffusion Probabilistic Models (DDPM)](#7-denoising-diffusion-probabilistic-models-ddpm)
8. [Stable Diffusion](#8-stable-diffusion)
9. [Conditional Generation](#9-conditional-generation)
10. [Evaluation Metrics](#10-evaluation-metrics)

---

## 1. Generative Models Fundamentals

### 1.1 What are Generative Models?

**Definition**: Model data distribution $P(X)$ to generate new samples.

**Types**:
- **Explicit density**: Model $P(X)$ directly (VAE, Flow-based)
- **Implicit density**: Sample without explicit $P(X)$ (GAN)
- **Iterative refinement**: Start from noise (Diffusion)

### 1.2 Applications

| Domain | Application |
|--------|-------------|
| **Vision** | Image generation, super-resolution, inpainting |
| **Language** | Text generation (GPT), translation |
| **Audio** | Speech synthesis, music generation |
| **Video** | Video synthesis, frame prediction |
| **Science** | Molecule design, protein folding |

### 1.3 Key Challenges

- **Mode collapse**: Generate limited variety
- **Evaluation**: How to measure quality?
- **Training stability**: GANs are notoriously unstable
- **Controllability**: Generate specific samples

---

## 2. Autoencoders

### 2.1 Architecture

```
Input → Encoder → Latent Code (z) → Decoder → Reconstruction
```

**Encoder**: $z = f_\theta(x)$  
**Decoder**: $\hat{x} = g_\phi(z)$

### 2.2 Loss Function

**Reconstruction loss**:
$$
\mathcal{L} = \| x - \hat{x} \|^2 = \| x - g_\phi(f_\theta(x)) \|^2
$$

### 2.3 Implementation

```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, latent_dim=32):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # For images in [0, 1]
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

# Training
model = Autoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for batch in dataloader:
    x, _ = batch
    x = x.view(x.size(0), -1)  # Flatten
    
    x_recon = model(x)
    loss = nn.MSELoss()(x_recon, x)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 2.4 Limitations

**Problem**: **Cannot generate new samples**!

Why? Latent space is not continuous/meaningful.

---

## 3. Variational Autoencoders (VAEs)

### 3.1 Motivation

**Goal**: Learn continuous latent space where we can sample.

**Idea**: Model latent as probability distribution.

### 3.2 Probabilistic Formulation

**Generative process**:
1. Sample latent: $z \sim P(z) = \mathcal{N}(0, I)$
2. Generate: $x \sim P_\theta(x | z)$

**Inference**: Given $x$, infer $z$ via $Q_\phi(z | x)$ (approximate posterior)

### 3.3 Evidence Lower Bound (ELBO)

**Log-likelihood** (intractable):
$$
\log P_\theta(x) = \log \int P_\theta(x | z) P(z) dz
$$

**ELBO** (variational lower bound):
$$
\log P_\theta(x) \geq \mathbb{E}_{z \sim Q_\phi(z|x)} [\log P_\theta(x | z)] - D_{\text{KL}}(Q_\phi(z|x) \| P(z))
$$

**Two terms**:
1. **Reconstruction**: $\mathbb{E}_{z \sim Q_\phi(z|x)} [\log P_\theta(x | z)]$
2. **Regularization**: $D_{\text{KL}}(Q_\phi(z|x) \| P(z))$

### 3.4 Reparameterization Trick

**Problem**: Cannot backprop through sampling.

**Solution**: Reparameterize
$$
z \sim \mathcal{N}(\mu, \sigma^2) \equiv z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

Now gradient can flow through $\mu, \sigma$.

### 3.5 VAE Loss

**Assuming Gaussian**:
- Encoder: $Q_\phi(z | x) = \mathcal{N}(z; \mu_\phi(x), \sigma_\phi^2(x))$
- Decoder: $P_\theta(x | z) = \mathcal{N}(x; \mu_\theta(z), \sigma_\theta^2 I)$

**KL divergence** (closed form for Gaussians):
$$
D_{\text{KL}}(Q_\phi(z|x) \| \mathcal{N}(0, I)) = \frac{1}{2} \sum_{j=1}^{J} \left( \mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1 \right)
$$

**Reconstruction loss**:
$$
\mathcal{L}_{\text{recon}} = \| x - \mu_\theta(z) \|^2
$$

**Total loss**:
$$
\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{\text{KL}}
$$

### 3.6 Implementation

```python
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    # Reconstruction loss
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss

# Training
model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for batch in dataloader:
    x, _ = batch
    x = x.view(x.size(0), -1)
    
    x_recon, mu, logvar = model(x)
    loss = vae_loss(x, x_recon, mu, logvar)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Generate new samples
with torch.no_grad():
    z = torch.randn(64, latent_dim)  # Sample from prior
    samples = model.decode(z)
```

### 3.7 β-VAE

**Idea**: Weight KL term with $\beta > 1$ for disentanglement.

$$
\mathcal{L}_{\beta\text{-VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{\text{KL}}
$$

**Effect**: Forces latent dimensions to be independent.

---

## 4. Generative Adversarial Networks (GANs)

### 4.1 Core Idea

**Two players**:
- **Generator** $G$: Create fake samples
- **Discriminator** $D$: Distinguish real vs fake

**Adversarial training**: Minimax game

### 4.2 Objective

**Discriminator** maximizes:
$$
\max_D \mathbb{E}_{x \sim P_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim P_z}[\log(1 - D(G(z)))]
$$

**Generator** minimizes:
$$
\min_G \mathbb{E}_{z \sim P_z}[\log(1 - D(G(z)))]
$$

**Combined** (minimax):
$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim P_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim P_z}[\log(1 - D(G(z)))]
$$

### 4.3 Training Algorithm

```
For each training iteration:
    1. Train Discriminator:
       - Sample real data: x ~ P_data
       - Sample noise: z ~ N(0, I)
       - Generate fake: x_fake = G(z)
       - Update D to maximize log D(x) + log(1 - D(x_fake))
    
    2. Train Generator:
       - Sample noise: z ~ N(0, I)
       - Update G to minimize log(1 - D(G(z)))
       (or equivalently maximize log D(G(z)))
```

### 4.4 Implementation

```python
class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_dim=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Training
G = Generator()
D = Discriminator()

optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCELoss()

for epoch in range(num_epochs):
    for real_images, _ in dataloader:
        batch_size = real_images.size(0)
        real_images = real_images.view(batch_size, -1)
        
        # Labels
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        # ========== Train Discriminator ==========
        # Real images
        outputs = D(real_images)
        d_loss_real = criterion(outputs, real_labels)
        
        # Fake images
        z = torch.randn(batch_size, latent_dim)
        fake_images = G(z)
        outputs = D(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()
        
        # ========== Train Generator ==========
        z = torch.randn(batch_size, latent_dim)
        fake_images = G(z)
        outputs = D(fake_images)
        
        # Generator wants D(G(z)) = 1
        g_loss = criterion(outputs, real_labels)
        
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
```

### 4.5 Common Problems

**1. Mode Collapse**:
- Generator produces limited variety
- Solution: Minibatch discrimination, unrolled GAN

**2. Vanishing Gradients**:
- When $D$ is too good, $G$ gets no signal
- Solution: Non-saturating loss, Wasserstein loss

**3. Training Instability**:
- Oscillations, divergence
- Solution: Careful hyperparameters, WGAN, spectral normalization

---

## 5. Advanced GAN Architectures

### 5.1 DCGAN (Deep Convolutional GAN)

**Key innovations**:
- Replace MLPs with CNNs
- Use strided convolutions (no pooling)
- Batch normalization
- LeakyReLU in D, ReLU in G

```python
class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, ngf=64):
        super().__init__()
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State: ngf x 32 x 32
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: 3 x 64 x 64
        )
    
    def forward(self, z):
        return self.main(z)

class DCGANDiscriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: ndf x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.main(x).view(-1, 1)
```

### 5.2 Wasserstein GAN (WGAN)

**Problem**: Original GAN loss unstable.

**Solution**: Use Wasserstein distance (Earth Mover's Distance).

**Objective**:
$$
\min_G \max_{D \in \mathcal{D}} \mathbb{E}_{x \sim P_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim P_z}[D(G(z))]
$$

Where $\mathcal{D}$ is set of 1-Lipschitz functions.

**Implementation**: Weight clipping
```python
# After updating D
for p in D.parameters():
    p.data.clamp_(-0.01, 0.01)  # Enforce Lipschitz
```

**WGAN-GP**: Use gradient penalty instead of clipping
$$
\mathcal{L}_D = -\mathbb{E}[D(x)] + \mathbb{E}[D(G(z))] + \lambda \mathbb{E}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]
$$

### 5.3 StyleGAN

**Key innovation**: Style-based generator

**Architecture**:
```
Latent z → Mapping Network → w
w → AdaIN layers → Image
```

**Adaptive Instance Normalization (AdaIN)**:
$$
\text{AdaIN}(x, y) = y_s \frac{x - \mu(x)}{\sigma(x)} + y_b
$$

Where $y_s, y_b$ come from $w$.

**Benefits**:
- Disentangled control
- Style mixing
- High quality (FID = 2.8 on FFHQ)

---

## 6. Diffusion Models

### 6.1 Core Idea

**Forward process**: Gradually add noise
$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
$$

**Reverse process**: Learn to denoise
$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

**Sampling**: Start from noise, iteratively denoise
$$
x_T \sim \mathcal{N}(0, I) \to x_{T-1} \to \cdots \to x_0
$$

### 6.2 Mathematical Formulation

**Forward closed form** (nice property):
$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)
$$

Where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$.

**Reparameterization**:
$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

---

## 7. Denoising Diffusion Probabilistic Models (DDPM)

### 7.1 Training Objective

**Simplified loss**:
$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

Where $\epsilon_\theta$ predicts the noise added at step $t$.

**Algorithm**:
```
1. Sample x_0 from dataset
2. Sample t ~ Uniform(1, T)
3. Sample noise ε ~ N(0, I)
4. Compute x_t = √(α̅_t) x_0 + √(1 - α̅_t) ε
5. Predict ε̂ = ε_θ(x_t, t)
6. Loss = || ε - ε̂ ||²
```

### 7.2 Implementation

```python
class DDPM(nn.Module):
    def __init__(self, noise_predictor, timesteps=1000):
        super().__init__()
        self.noise_predictor = noise_predictor  # UNet
        self.timesteps = timesteps
        
        # Define beta schedule
        self.betas = torch.linspace(1e-4, 0.02, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def forward(self, x_0):
        # Sample timestep
        t = torch.randint(0, self.timesteps, (x_0.size(0),))
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Compute x_t
        sqrt_alpha_cumprod_t = self.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        
        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
        
        # Predict noise
        noise_pred = self.noise_predictor(x_t, t)
        
        # Loss
        loss = F.mse_loss(noise_pred, noise)
        return loss
    
    @torch.no_grad()
    def sample(self, shape):
        # Start from pure noise
        x = torch.randn(shape)
        
        # Iteratively denoise
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((shape[0],), t, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.noise_predictor(x, t_batch)
            
            # Compute mean
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            
            mean = (x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * noise_pred) / torch.sqrt(alpha_t)
            
            # Add noise (except last step)
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean
        
        return x
```

### 7.3 DDIM (Faster Sampling)

**Problem**: DDPM requires 1000 steps (slow!).

**Solution**: Non-Markovian process with fewer steps.

**Sampling** (deterministic):
$$
x_{t-\Delta t} = \sqrt{\bar{\alpha}_{t-\Delta t}} \left( \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1 - \bar{\alpha}_{t-\Delta t}} \epsilon_\theta(x_t, t)
$$

**Result**: 50 steps instead of 1000 (20× faster)!

---

## 8. Stable Diffusion

### 8.1 Latent Diffusion

**Idea**: Run diffusion in latent space (not pixel space).

**Architecture**:
```
Image → VAE Encoder → Latent (z)
    ↓
Diffusion on z
    ↓
Latent (z) → VAE Decoder → Image
```

**Benefits**:
- **Efficiency**: $512 \times 512 \times 3$ → $64 \times 64 \times 4$ (192× fewer values)
- **Quality**: Smooth latent space
- **Speed**: Training and inference faster

### 8.2 Conditioning (Text-to-Image)

**Cross-attention** in UNet:
```python
class CrossAttnBlock(nn.Module):
    def __init__(self, dim, context_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=8)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(context_dim, dim)
        self.v_proj = nn.Linear(context_dim, dim)
    
    def forward(self, x, context):
        # x: image features [B, N, D]
        # context: text features [B, M, C]
        
        # Cross-attention
        Q = self.q_proj(self.norm1(x))
        K = self.k_proj(context)
        V = self.v_proj(context)
        
        attn_out, _ = self.attn(Q, K, V)
        x = x + attn_out
        
        return self.norm2(x)
```

### 8.3 Classifier-Free Guidance

**Technique**: Amplify conditional signal.

**Sampling**:
$$
\epsilon_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + s \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))
$$

Where:
- $c$ = condition (text)
- $\emptyset$ = unconditional
- $s$ = guidance scale (typically 7.5)

**Effect**: Higher $s$ = more adherence to prompt (but less diversity).

---

## 9. Conditional Generation

### 9.1 Class-Conditional Generation

**Discriminator** sees both image and label:
$$
D(x, y)
$$

**Generator** takes noise + label:
$$
G(z, y)
$$

### 9.2 Image-to-Image Translation

**Pix2Pix**: Paired translation (e.g., edges → photo)

**CycleGAN**: Unpaired translation
- Two generators: $G: X \to Y$, $F: Y \to X$
- Cycle consistency: $F(G(x)) \approx x$

---

## 10. Evaluation Metrics

### 10.1 Inception Score (IS)

**Idea**: Good samples should be:
1. **Confident**: Low entropy $p(y | x)$
2. **Diverse**: High entropy $p(y)$

$$
\text{IS} = \exp\left( \mathbb{E}_{x \sim p_g} D_{\text{KL}}(p(y|x) \| p(y)) \right)
$$

### 10.2 Fréchet Inception Distance (FID)

**Idea**: Distance between real and generated distributions in feature space.

```python
def calculate_fid(real_images, generated_images, inception_model):
    # Extract features
    real_features = inception_model(real_images)
    gen_features = inception_model(generated_images)
    
    # Compute mean and covariance
    mu_real, sigma_real = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = gen_features.mean(0), np.cov(gen_features, rowvar=False)
    
    # Fréchet distance
    diff = mu_real - mu_gen
    covmean = scipy.linalg.sqrtm(sigma_real @ sigma_gen)
    
    fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid
```

**Lower is better** (0 = identical distributions).

---

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [Autoencoder](notebooks/01_autoencoder.ipynb) | Build basic autoencoder |
| 2 | [VAE](notebooks/02_vae.ipynb) | Variational autoencoder with ELBO |
| 3 | [GAN](notebooks/03_gan.ipynb) | Vanilla GAN on MNIST |
| 4 | [DCGAN](notebooks/04_dcgan.ipynb) | Deep convolutional GAN |
| 5 | [DDPM](notebooks/05_ddpm.ipynb) | Diffusion model from scratch |
| 6 | [Stable Diffusion](notebooks/06_stable_diffusion.ipynb) | Text-to-image generation |

---

## Projects

### Mini Project: Face Generator
- Train DCGAN on CelebA dataset
- Generate realistic faces
- Explore latent space interpolation
- Compare with VAE

### Advanced Project: Custom Stable Diffusion
- Fine-tune Stable Diffusion on custom domain
- Add LoRA for personalization
- Build Gradio interface
- Compare FID scores

---

## Interview Questions

1. Explain the difference between VAE and GAN.
2. Derive the VAE loss (ELBO).
3. What is the reparameterization trick and why is it needed?
4. Walk through GAN training algorithm.
5. What is mode collapse and how do you fix it?
6. Explain WGAN and why Wasserstein distance is better.
7. How does diffusion differ from GANs/VAEs?
8. Walk through DDPM training and sampling.
9. What is latent diffusion and why is it efficient?
10. How does classifier-free guidance work in Stable Diffusion?
