from graddy.nn import MLP
from graddy.engine import Value

# 1. Define a simple dataset (binary classification)
# 4 samples, each with 3 input features
inputs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
# Desired targets
ys = [1.0, -1.0, -1.0, 1.0]

# 2. Define a Multilayer Perceptron
# Input layer size: 3
# Hidden layers: [8, 8]
# Output layer size: 1
model = MLP(3, [8, 8, 1])
print("Initial model architecture:")
print(model)
print(f"Number of parameters: {len(model.parameters())}")

# 3. Simple training loop (Gradient Descent)
print("\nTraining MLP...")
epochs = 100
learning_rate = 0.01

for k in range(epochs):
    # Forward pass
    ypred = [model(x) for x in inputs]
    
    # Calculate MSE loss
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    # Reset gradients
    model.zero_grad()
    
    # Backward pass
    loss.backward()
    
    # Update weights/biases
    for p in model.parameters():
        p.data -= learning_rate * p.grad
        
    if (k + 1) % 10 == 0 or k == 0:
        print(f"Epoch {k+1:2d} | Loss: {loss.data:.6f}")

print("\nFinal predictions:")
ypred = [model(x) for x in inputs]
for x, ygt, ypred_val in zip(inputs, ys, ypred):
    print(f"Input: {x} | Target: {ygt:4.1f} | Prediction: {ypred_val.data: .4f}")
