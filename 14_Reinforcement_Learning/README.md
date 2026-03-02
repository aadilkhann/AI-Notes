# Module 14: Reinforcement Learning

> **Level**: Advanced  
> **Duration**: 4–5 weeks  
> **Prerequisites**: Modules 03 (Deep Learning), 08 (LLMs for RLHF)  
> **Goal**: Master RL from fundamentals to RLHF for LLMs

---

## Table of Contents

1. [Reinforcement Learning Fundamentals](#1-reinforcement-learning-fundamentals)
2. [Markov Decision Processes (MDPs)](#2-markov-decision-processes-mdps)
3. [Value-Based Methods](#3-value-based-methods)
4. [Policy-Based Methods](#4-policy-based-methods)
5. [Actor-Critic Methods](#5-actor-critic-methods)
6. [Deep RL: DQN](#6-deep-rl-dqn)
7. [Policy Gradient Algorithms](#7-policy-gradient-algorithms)
8. [Proximal Policy Optimization (PPO)](#8-proximal-policy-optimization-ppo)
9. [RLHF for LLMs](#9-rlhf-for-llms)
10. [Advanced RL Topics](#10-advanced-rl-topics)

---

## 1. Reinforcement Learning Fundamentals

### 1.1 What is Reinforcement Learning?

**Definition**: Learning through interaction to maximize cumulative reward.

**Key components**:
- **Agent**: Learner/decision-maker
- **Environment**: What agent interacts with
- **State** $s$: Current situation
- **Action** $a$: What agent can do
- **Reward** $r$: Feedback signal

### 1.2 RL vs Other Learning Paradigms

| Paradigm | Feedback | Data |
|----------|----------|------|
| **Supervised** | (x, y) labels | i.i.d. |
| **Unsupervised** | No labels | i.i.d. |
| **RL** | Delayed rewards | Sequential, correlated |

### 1.3 The RL Framework

```
Agent observes state s_t
    ↓
Agent takes action a_t
    ↓
Environment returns reward r_t and next state s_{t+1}
    ↓
Repeat
```

**Goal**: Learn policy $\pi(a | s)$ to maximize expected return.

### 1.4 Key Challenges

- **Credit assignment**: Which action caused reward?
- **Exploration vs exploitation**: Try new things vs use known good actions
- **Delayed rewards**: Reward may come many steps later
- **Non-stationarity**: Agent's actions change data distribution

---

## 2. Markov Decision Processes (MDPs)

### 2.1 Formal Definition

**MDP** = $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$

- $\mathcal{S}$: State space
- $\mathcal{A}$: Action space
- $P(s' | s, a)$: Transition probability
- $R(s, a)$: Reward function
- $\gamma \in [0, 1]$: Discount factor

### 2.2 Markov Property

**Definition**: Future depends only on current state, not history.
$$
P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} | s_t, a_t)
$$

### 2.3 Return and Value Functions

**Return** (cumulative discounted reward):
$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
$$

**State-value function**:
$$
V^\pi(s) = \mathbb{E}_\pi[G_t | s_t = s]
$$

**Action-value function** (Q-function):
$$
Q^\pi(s, a) = \mathbb{E}_\pi[G_t | s_t = s, a_t = a]
$$

### 2.4 Bellman Equations

**Bellman expectation equation**:
$$
V^\pi(s) = \sum_{a} \pi(a | s) \sum_{s'} P(s' | s, a) [R(s, a) + \gamma V^\pi(s')]
$$

**Bellman optimality equation**:
$$
V^*(s) = \max_a \sum_{s'} P(s' | s, a) [R(s, a) + \gamma V^*(s')]
$$

$$
Q^*(s, a) = \sum_{s'} P(s' | s, a) [R(s, a) + \gamma \max_{a'} Q^*(s', a')]
$$

---

## 3. Value-Based Methods

### 3.1 Dynamic Programming

**Policy Evaluation**: Compute $V^\pi(s)$
```
Initialize V(s) = 0 for all s
Repeat until convergence:
    For each state s:
        V(s) ← Σ_a π(a|s) Σ_s' P(s'|s,a) [R(s,a) + γ V(s')]
```

**Value Iteration**: Find $V^*(s)$
```
Initialize V(s) = 0 for all s
Repeat until convergence:
    For each state s:
        V(s) ← max_a Σ_s' P(s'|s,a) [R(s,a) + γ V(s')]
```

### 3.2 Monte Carlo Methods

**Idea**: Learn from complete episodes.

**First-visit MC**:
```
For each episode:
    For each state s visited in episode:
        G ← return following first visit to s
        V(s) ← V(s) + α [G - V(s)]
```

**No model needed**: Don't need to know $P(s' | s, a)$.

### 3.3 Temporal Difference (TD) Learning

**Idea**: Bootstrap from current estimate.

**TD(0) update**:
$$
V(s_t) \leftarrow V(s_t) + \alpha [r_t + \gamma V(s_{t+1}) - V(s_t)]
$$

Where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is **TD error**.

**Benefits**:
- Learn online (no need to wait for episode end)
- Lower variance than MC
- Bootstrapping enables learning

### 3.4 Q-Learning

**Off-policy TD control**:
$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]
$$

**Algorithm**:
```python
import numpy as np

class QLearning:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def choose_action(self, state):
        # ε-greedy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target += self.gamma * np.max(self.Q[next_state])
        
        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

# Training
env = gym.make('FrozenLake-v1')
agent = QLearning(env.observation_space.n, env.action_space.n)

for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
```

### 3.5 SARSA (On-Policy)

**Difference**: Use actual next action (not max)
$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]
$$

**Q-learning vs SARSA**:
- Q-learning: Off-policy (learns optimal policy)
- SARSA: On-policy (learns policy being followed)

---

## 4. Policy-Based Methods

### 4.1 Policy Representation

**Parameterized policy**: $\pi_\theta(a | s)$

**For discrete actions**:
$$
\pi_\theta(a | s) = \frac{\exp(h_\theta(s, a))}{\sum_{a'} \exp(h_\theta(s, a'))}
$$

**For continuous actions**:
$$
\pi_\theta(a | s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta^2)
$$

### 4.2 Policy Gradient Theorem

**Objective**: Maximize expected return
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)]
$$

**Policy gradient** (REINFORCE):
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ G_t \nabla_\theta \log \pi_\theta(a_t | s_t) \right]
$$

**Intuition**: Increase probability of actions that led to high return.

### 4.3 REINFORCE Algorithm

```python
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.network(state)

class REINFORCE:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
    
    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.policy(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)
    
    def update(self, log_probs, rewards):
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize
        
        # Policy gradient
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Training
env = gym.make('CartPole-v1')
agent = REINFORCE(state_dim=4, action_dim=2)

for episode in range(1000):
    state = env.reset()
    log_probs = []
    rewards = []
    
    done = False
    while not done:
        action, log_prob = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        log_probs.append(log_prob)
        rewards.append(reward)
        state = next_state
    
    agent.update(log_probs, rewards)
```

### 4.4 Baseline

**Problem**: High variance in policy gradient.

**Solution**: Subtract baseline $b(s)$
$$
\nabla_\theta J(\theta) = \mathbb{E} \left[ (G_t - b(s_t)) \nabla_\theta \log \pi_\theta(a_t | s_t) \right]
$$

**Common choice**: $b(s) = V(s)$ (state value)

---

## 5. Actor-Critic Methods

### 5.1 Motivation

**Combine**:
- **Actor**: Policy $\pi_\theta(a | s)$
- **Critic**: Value function $V_\phi(s)$ or $Q_\phi(s, a)$

**Advantage**: Lower variance than REINFORCE.

### 5.2 Advantage Function

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

**Interpretation**: How much better is action $a$ than average?

**Estimate**:
$$
\hat{A}_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

### 5.3 Actor-Critic Algorithm

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        shared_features = self.shared(state)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_probs, state_value

class A2C:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
    
    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Forward pass
        action_probs, state_values = self.model(states)
        _, next_state_values = self.model(next_states)
        
        # Compute advantages
        td_targets = rewards + self.gamma * next_state_values.squeeze() * (1 - dones)
        advantages = td_targets - state_values.squeeze()
        
        # Actor loss
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss
        critic_loss = advantages.pow(2).mean()
        
        # Total loss
        loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

---

## 6. Deep RL: DQN

### 6.1 Deep Q-Network

**Idea**: Use neural network to approximate $Q(s, a)$.

**Challenges**:
1. **Correlated samples**: Sequential data breaks i.i.d. assumption
2. **Non-stationary targets**: Q-values keep changing

### 6.2 Key Innovations

**Experience Replay**:
- Store transitions $(s, a, r, s')$ in replay buffer
- Sample random minibatches for training
- Breaks correlations

**Target Network**:
- Use separate network $Q_{\theta^-}$ for targets
- Update slowly: $\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-$
- Stabilizes training

### 6.3 DQN Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(states),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(next_states),
                torch.FloatTensor(dones))
    
    def __len__(self):
        return len(self.buffer)

class DQN:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0):
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.q_network.network[-1].out_features - 1)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state)
                return q_values.argmax().item()
    
    def update(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Current Q values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Loss
        loss = nn.MSELoss()(q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
```

### 6.4 Double DQN

**Problem**: DQN overestimates Q-values.

**Solution**: Decouple action selection and evaluation
$$
Q_{\text{target}} = r + \gamma Q_{\theta^-}(s', \arg\max_{a'} Q_\theta(s', a'))
$$

---

## 7. Policy Gradient Algorithms

### 7.1 Trust Region Policy Optimization (TRPO)

**Problem**: Too large policy update can be catastrophic.

**Solution**: Constrain KL divergence
$$
\max_\theta \mathbb{E}[\text{Advantage}] \quad \text{s.t. } \mathbb{E}[D_{\text{KL}}(\pi_{\theta_{\text{old}}} \| \pi_\theta)] \leq \delta
$$

**Implementation**: Complex (requires conjugate gradient).

---

## 8. Proximal Policy Optimization (PPO)

### 8.1 Motivation

**Goal**: TRPO's benefits without complexity.

### 8.2 Clipped Objective

**Probability ratio**:
$$
r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}
$$

**Clipped surrogate objective**:
$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]
$$

**Intuition**: Limit how much policy can change.

### 8.3 PPO Implementation

```python
class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, epochs=10):
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
    
    def compute_gae(self, rewards, values, next_values, dones, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages)
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        for _ in range(self.epochs):
            # Get current policy
            action_probs, state_values = self.actor_critic(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            
            # Actor loss
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = nn.MSELoss()(state_values.squeeze(), returns)
            
            # Total loss
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()
```

### 8.4 PPO Training Loop

```python
def train_ppo(env, agent, num_episodes=1000):
    for episode in range(num_episodes):
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        
        state = env.reset()
        done = False
        
        # Collect rollout
        while not done:
            action_probs, value = agent.actor_critic(torch.FloatTensor(state))
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            
            next_state, reward, done, _ = env.step(action.item())
            
            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(dist.log_prob(action).item())
            values.append(value.item())
            dones.append(done)
            
            state = next_state
        
        # Compute returns and advantages
        _, next_values = agent.actor_critic(torch.FloatTensor(next_state))
        advantages = agent.compute_gae(rewards, values, [next_values.item()], dones)
        returns = advantages + torch.tensor(values)
        
        # Update policy
        agent.update(states, actions, log_probs, returns, advantages)
```

---

## 9. RLHF for LLMs

### 9.1 Why RLHF?

**Problem**: LLMs trained on next-token prediction don't align with human preferences.

**Examples**:
- Toxic or biased outputs
- Unhelpful responses
- Hallucinations

**Solution**: Optimize for human feedback.

### 9.2 Three-Stage Process

**Stage 1: Supervised Fine-Tuning (SFT)**
- Collect high-quality demonstrations
- Fine-tune base model

**Stage 2: Reward Modeling**
- Collect preference data: $(x, y_w, y_l)$
- Train reward model: $r_\phi(x, y)$

**Stage 3: RL Optimization**
- Use PPO to maximize reward
- Add KL penalty to prevent drift

### 9.3 Reward Model

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        # Use last token
        last_hidden = outputs.last_hidden_state[:, -1, :]
        reward = self.value_head(last_hidden)
        return reward

def train_reward_model(model, preference_data):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    for batch in preference_data:
        prompt, winner, loser = batch
        
        # Concatenate prompt + completion
        winner_input = tokenizer(prompt + winner, return_tensors="pt")
        loser_input = tokenizer(prompt + loser, return_tensors="pt")
        
        # Get rewards
        r_winner = model(**winner_input)
        r_loser = model(**loser_input)
        
        # Bradley-Terry loss
        loss = -torch.log(torch.sigmoid(r_winner - r_loser)).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 9.4 PPO for Language

**Modified objective**:
$$
J(\theta) = \mathbb{E}_{x, y \sim \pi_\theta} \left[ r_\phi(x, y) - \beta \log \frac{\pi_\theta(y | x)}{\pi_{\text{ref}}(y | x)} \right]
$$

**KL penalty**: Prevents model from drifting too far from SFT model.

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# Load models
model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
ref_model = AutoModelForCausalLM.from_pretrained("gpt2")
reward_model = RewardModel.from_pretrained("reward_model")

# PPO config
ppo_config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
    ppo_epochs=4,
)

# Trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# Training loop
for query in prompts:
    # Generate response
    response = ppo_trainer.generate(query)
    
    # Get reward
    reward = reward_model(query, response)
    
    # PPO update
    stats = ppo_trainer.step([query], [response], [reward])
```

---

## 10. Advanced RL Topics

### 10.1 Multi-Agent RL

**Challenges**:
- Non-stationary environment (other agents learning)
- Credit assignment
- Communication

**Algorithms**:
- Independent Q-learning
- QMIX
- MADDPG

### 10.2 Offline RL

**Problem**: Learn from fixed dataset (no exploration).

**Applications**:
- Healthcare (can't experiment on patients)
- Robotics (expensive data collection)
- LLMs (learn from static datasets)

**Algorithms**:
- Conservative Q-Learning (CQL)
- Implicit Q-Learning (IQL)

### 10.3 Model-Based RL

**Idea**: Learn model of environment, use for planning.

**Algorithms**:
- Dyna-Q
- MuZero
- Dreamer

---

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [Q-Learning](notebooks/01_q_learning.ipynb) | Tabular Q-learning on GridWorld |
| 2 | [REINFORCE](notebooks/02_reinforce.ipynb) | Policy gradient on CartPole |
| 3 | [Actor-Critic](notebooks/03_actor_critic.ipynb) | A2C algorithm |
| 4 | [DQN](notebooks/04_dqn.ipynb) | Deep Q-Network on Atari |
| 5 | [PPO](notebooks/05_ppo.ipynb) | PPO from scratch |
| 6 | [RLHF](notebooks/06_rlhf.ipynb) | Fine-tune LLM with RLHF |

---

## Projects

### Mini Project: Train CartPole Agent
- Implement DQN from scratch
- Compare with Q-learning and PPO
- Visualize learning curves and policies
- Analyze exploration strategies

### Advanced Project: RLHF Pipeline
- Fine-tune GPT-2 on helpfulness
- Collect synthetic preference data with GPT-4
- Train reward model
- Run PPO optimization
- Evaluate with human eval

---

## Interview Questions

1. Explain the difference between on-policy and off-policy methods.
2. Derive the policy gradient theorem.
3. What is the credit assignment problem in RL?
4. Why does DQN use experience replay and target networks?
5. Walk through the PPO objective and why clipping helps.
6. Explain the three stages of RLHF.
7. What is the KL penalty in RLHF and why is it needed?
8. Compare value-based and policy-based methods.
9. What is the exploration-exploitation tradeoff?
10. How does the reward model in RLHF work?
