# Graddy

![Graddy Banner](banner.jpg)

**Graddy** is a minimalist, scalar-valued autograd (automatic differentiation) engine and neural network library written in Python. It is designed for educational purposes, providing a clean and intuitive clone of Andrej Karpathy's `micrograd`.

Graddy compiles a Directed Acyclic Graph (DAG) of mathematical operations dynamically during the forward pass and uses backpropagation (automatic differentiation) over the topological sort of the graph to compute gradients.

---

## Features

- **Scalar Autograd Engine**: Implemented via the `Value` class, supporting:
  - Basic arithmetic: addition, subtraction, multiplication, division, negation, and powers.
  - Activation and math functions: `relu()`, `exp()`, and `tanh()`.
- **Neural Network Layer**: Implements core deep learning layers using scalar math:
  - `Neuron`: Custom scalar neuron with weights and bias.
  - `Layer`: A collection of neurons.
  - `MLP`: Multi-Layer Perceptron.
- **Reference Validation**: Built-in test suite verifying forward and backward pass gradients against PyTorch.

---

## Installation & Setup

1. **Activate Virtual Environment**:
   ```bash
   .venv\Scripts\activate
   ```
2. **Install Dependencies** (if needed, e.g., PyTorch for running the verification tests):
   ```bash
   pip install torch
   ```

---

## Usage Examples

### 1. Basic Autograd Operations
```python
from graddy.engine import Value

# Define input scalars
x = Value(-4.0)
z = 2 * x + 2 + x
q = z.relu() + z * x
h = (z * z).relu()
y = h + q + q * x

# Run backward pass (backpropagation)
y.backward()

print(f"y: {y.data}")  # The outcome of the forward pass
print(f"x gradient: {x.grad}")  # The derivative dy/dx
```

### 2. Training a Multi-Layer Perceptron (MLP)
You can build and train a multi-layer neural network on a custom binary classification dataset:
```python
from graddy.nn import MLP

# 4 data points with 3 input features
inputs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0] # Target labels

# Initialize MLP: 3 inputs -> hidden layer of 8 -> hidden layer of 8 -> 1 output
model = MLP(3, [8, 8, 1])

# Training loop
for epoch in range(300):
    # Forward pass
    ypred = [model(x) for x in inputs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    # Zero gradients
    model.zero_grad()
    
    # Backward pass
    loss.backward()
    
    # Gradient descent step
    for p in model.parameters():
        p.data -= 0.01 * p.grad
        
    print(f"Epoch {epoch+1} | Loss: {loss.data:.4f}")
```

---

## Running the Demo & Tests

### Running the Demos
Three demonstration scripts are provided in the project root:
1. **Basic MLP Classification Demo**:
   ```bash
   python demo.py
   ```
2. **Moons Dataset Classification & Plotting Demo** (requires `scikit-learn` and `matplotlib`):
   ```bash
   python moons_demo.py
   ```
3. **Neural Network Architecture Plotting Demo** (requires `scikit-learn` and `matplotlib`):
   ```bash
   python plot_nn.py
   ```

### Running the Tests
To verify all operations and gradients against PyTorch references:
```bash
python test/test.py
```

---

## 🗺️ Roadmap

Future plans and feature implementations:
- [ ] **Tensor Support**: Transition from scalar-valued `Value` units to multidimensional tensors.
- [ ] **Broadcasting**: Element-wise operations across tensor shapes of different sizes.
- [ ] **Matrix Multiplication**: Optimized matrix multiply ($X \cdot W$) operations for faster layer execution.
- [ ] **Softmax & Cross-Entropy Loss**: Standard classification metrics and loss components.
- [ ] **Adam Optimizer**: Momentum-based adaptive learning rate optimizer for faster convergence.
- [ ] **Gradient Checking**: Finite-difference approximations to numerically verify gradient values.
- [ ] **DataLoader**: Mini-batch processing, shuffling, and data pipeline management.
- **Advanced Architectures**:
  - [ ] **CNN Layers**: 2D convolution and max-pooling operations.
  - [ ] **Transformer Blocks**: Multi-head self-attention mechanisms and layer normalization.
- [ ] **GPU Backend**: Integration with CUDA or OpenCL for hardware-accelerated tensor computations.

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).
