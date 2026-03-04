# Mini Project: MNIST Digit Classifier from Scratch

## Overview

Build a complete neural network from scratch (only NumPy) to classify handwritten digits from the MNIST dataset.

**Goal**: Achieve **>95% test accuracy** using only NumPy (no PyTorch/TensorFlow).

## Learning Objectives

1. Implement forward and backward propagation from scratch
2. Understand weight initialization strategies
3. Implement mini-batch gradient descent
4. Add regularization techniques
5. Debug training dynamics

## Project Structure

```
mini_project/
├── README.md (this file)
├── mnist_classifier.py      # Your implementation
├── utils.py                 # Helper functions
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
└── requirements.txt
```

## Requirements

```txt
numpy>=1.24
matplotlib>=3.7
scikit-learn>=1.3
pillow>=10.0
```

## Part 1: Data Loading (10 points)

**Task**: Load and preprocess MNIST dataset

```python
from sklearn.datasets import fetch_openml
import numpy as np

def load_mnist():
    """Load MNIST dataset and preprocess.
    
    Returns:
        X_train: (60000, 784) normalized to [0, 1]
        y_train: (60000, 10) one-hot encoded
        X_test: (10000, 784)
        y_test: (10000, 10)
    """
    # TODO: Implement this
    pass

# Preprocessing steps:
# 1. Load data using fetch_openml('mnist_784', parser='auto')
# 2. Normalize pixel values: X / 255.0
# 3. One-hot encode labels
# 4. Shuffle training data
```

**Deliverables**:
- [ ] Load MNIST (70,000 images)
- [ ] Normalize pixels to [0, 1]
- [ ] One-hot encode labels
- [ ] Split into train/validation/test (50k/10k/10k)

## Part 2: Network Architecture (20 points)

**Task**: Implement a 3-layer neural network

```
Architecture: 784 → 128 → 64 → 10
Activations:  ReLU   ReLU   Softmax
```

```python
class NeuralNetwork:
    def __init__(self, layer_dims=[784, 128, 64, 10]):
        """Initialize network with He initialization."""
        self.layer_dims = layer_dims
        self.L = len(layer_dims) - 1
        self.parameters = {}
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """He initialization for ReLU networks."""
        # TODO: Implement He initialization
        # Var(W) = 2 / n_in
        pass
    
    def forward(self, X):
        """Forward pass through all layers."""
        # TODO: Implement forward pass
        # Store Z and A in cache for backprop
        pass
    
    def backward(self, Y):
        """Backpropagation to compute gradients."""
        # TODO: Implement backprop
        pass
    
    def update(self, grads, learning_rate):
        """Update parameters using gradient descent."""
        # TODO: Implement parameter update
        pass
```

**Deliverables**:
- [ ] Initialize weights with He initialization
- [ ] Implement forward pass with caching
- [ ] Implement backward pass
- [ ] Implement parameter updates

## Part 3: Training Loop (20 points)

**Task**: Implement mini-batch gradient descent with learning rate decay

```python
def train(model, X_train, y_train, X_val, y_val, 
          epochs=50, batch_size=128, learning_rate=0.1):
    """Training loop with mini-batch SGD.
    
    Features:
    - Mini-batch gradient descent
    - Learning rate decay
    - Early stopping
    - Progress tracking
    """
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Forward pass
            # Backward pass
            # Update parameters
            pass
        
        # Validation
        # Learning rate decay
        # Early stopping
        
    return train_losses, val_losses, val_accuracies
```

**Deliverables**:
- [ ] Mini-batch SGD implementation
- [ ] Learning rate decay schedule
- [ ] Validation monitoring
- [ ] Early stopping logic

## Part 4: Regularization (20 points)

**Task**: Add L2 regularization and dropout

### L2 Regularization

```python
def compute_loss(self, Y, Y_hat, lambda_reg=0.01):
    """Cross-entropy loss + L2 regularization.
    
    L = CrossEntropy + (λ/2m) * Σ||W||²
    """
    m = Y.shape[0]
    
    # Cross-entropy
    ce_loss = -np.mean(np.sum(Y * np.log(Y_hat + 1e-8), axis=1))
    
    # L2 regularization
    l2_loss = 0
    for l in range(1, self.L + 1):
        l2_loss += np.sum(self.parameters[f'W{l}'] ** 2)
    
    l2_loss *= (lambda_reg / (2 * m))
    
    return ce_loss + l2_loss
```

### Dropout

```python
def forward_with_dropout(self, X, keep_prob=0.8):
    """Forward pass with dropout regularization."""
    # TODO: Implement dropout
    # During training: randomly zero out neurons
    # During inference: scale activations by keep_prob
    pass
```

**Deliverables**:
- [ ] L2 regularization in loss
- [ ] L2 gradients in backprop
- [ ] Dropout during training
- [ ] Proper inference (no dropout)

## Part 5: Evaluation & Visualization (15 points)

**Task**: Evaluate model and visualize results

```python
def evaluate(model, X_test, y_test):
    """Comprehensive evaluation."""
    y_pred = model.predict(X_test)
    accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))
    
    # Confusion matrix
    # Per-class accuracy
    # Misclassified examples
    
    return metrics

def plot_training_curves(train_losses, val_losses, val_accuracies):
    """Plot training dynamics."""
    # TODO: Create 2 subplots
    # 1. Train/Val loss over time
    # 2. Val accuracy over time
    pass

def visualize_predictions(model, X_test, y_test, num_samples=16):
    """Show predictions with confidence."""
    # TODO: Display grid of predictions
    # Show: image, true label, predicted label, confidence
    pass
```

**Deliverables**:
- [ ] Test accuracy computation
- [ ] Confusion matrix
- [ ] Training curves plot
- [ ] Prediction visualization
- [ ] Error analysis (which digits are confused?)

## Part 6: Hyperparameter Tuning (15 points)

**Task**: Experiment with different hyperparameters

| Hyperparameter | Options to Try |
|----------------|----------------|
| **Architecture** | [784, 256, 10], [784, 512, 256, 10], [784, 128, 64, 32, 10] |
| **Learning Rate** | 0.01, 0.05, 0.1, 0.5 |
| **Batch Size** | 32, 64, 128, 256 |
| **Regularization** | λ = 0, 0.001, 0.01, 0.1 |
| **Dropout** | keep_prob = 0.7, 0.8, 0.9, 1.0 |
| **Initialization** | Xavier, He, Random small |

**Deliverables**:
- [ ] Train at least 5 different configurations
- [ ] Compare results in a table
- [ ] Plot performance comparison
- [ ] Document best configuration

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| Data Loading | 10 | Correct preprocessing, shuffling, splitting |
| Architecture | 20 | Clean implementation, proper shapes |
| Training Loop | 20 | Mini-batch SGD, learning rate decay |
| Regularization | 20 | L2 + Dropout correctly implemented |
| Evaluation | 15 | Comprehensive metrics and visualizations |
| Hyperparameter Tuning | 15 | Systematic experiments, documented results |
| **Total** | **100** | **>95% test accuracy = Full marks** |

## Bonus Challenges (+20 points)

1. **Advanced Optimization** (+5): Implement Momentum or Adam optimizer
2. **Batch Normalization** (+5): Add batch norm layers
3. **Data Augmentation** (+5): Random rotation, translation
4. **Custom Loss** (+5): Implement focal loss for hard examples

## Expected Results

With proper implementation, you should achieve:

- **Training accuracy**: ~98-99%
- **Validation accuracy**: ~97-98%
- **Test accuracy**: ~96-97%
- **Training time**: ~5-10 minutes on CPU

## Common Pitfalls & Debugging Tips

### Issue: Loss is NaN
- **Cause**: Numerical instability in log or exp
- **Fix**: Clip values, add epsilon to log

### Issue: Accuracy stuck at ~10% (random guessing)
- **Cause**: All gradients are zero
- **Fix**: Check initialization (avoid zeros), check gradient computation

### Issue: Training loss decreases but val loss increases
- **Cause**: Overfitting
- **Fix**: Add regularization, reduce model size, add dropout

### Issue: Very slow convergence
- **Cause**: Learning rate too small, poor initialization
- **Fix**: Increase learning rate, use He initialization

### Issue: Loss oscillates wildly
- **Cause**: Learning rate too large
- **Fix**: Reduce learning rate, use learning rate decay

## Debugging Checklist

- [ ] Check data shapes at each layer
- [ ] Verify forward pass with known input
- [ ] Gradient checking (numerical vs backprop)
- [ ] Sanity check: Overfit on small batch (should reach 100%)
- [ ] Visualize weight distributions
- [ ] Monitor gradient magnitudes

## Submission

Submit a single Python script `mnist_classifier.py` that:

```bash
$ python mnist_classifier.py
Loading MNIST...
Training...
Epoch   10 | Loss: 0.234 | Val Acc: 94.2%
Epoch   20 | Loss: 0.156 | Val Acc: 95.8%
Epoch   30 | Loss: 0.123 | Val Acc: 96.4%
...
Final Test Accuracy: 96.7%
```

Include:
1. Your implementation
2. Training logs
3. Plots of training curves
4. Confusion matrix
5. Analysis document (1-2 pages)

## Timeline

- **Week 1**: Parts 1-2 (Data + Architecture)
- **Week 2**: Parts 3-4 (Training + Regularization)
- **Week 3**: Parts 5-6 (Evaluation + Tuning)

## Resources

- [Deep Learning Book - Chapter 6 (Backprop)](http://www.deeplearningbook.org/)
- [CS231n Notes on Backprop](http://cs231n.github.io/optimization-2/)
- [Yes You Should Understand Backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)

## Success Criteria

✅ **Minimum**: 95% test accuracy with clean code  
🌟 **Excellent**: 97%+ accuracy with all bonus features  
🏆 **Outstanding**: 98%+ accuracy, comprehensive analysis, novel techniques

Good luck! Remember: The goal is not just to get high accuracy, but to **deeply understand** how neural networks work.
