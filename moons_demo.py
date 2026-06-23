import sys
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from graddy.nn import MLP

# Set seed for reproducibility
random.seed(1337)
np.random.seed(1337)

# 1. Generate the moons dataset
X, y = make_moons(n_samples=100, noise=0.1)
y = y * 2 - 1 # convert labels from 0/1 to -1/1

# 2. Initialize the MLP model
# 2 inputs (x, y coordinates), 2 hidden layers of 16 neurons, 1 output
model = MLP(2, [16, 16, 1])
print(f"Initialized MLP model with {len(model.parameters())} parameters.")

# 3. Training Loop with SVM Hinge Loss and L2 Regularization
epochs = 100
learning_rate = 1.0
regularization_strength = 1e-4

print("Training MLP on Moons classification task...")
for epoch in range(epochs):
    # Forward pass: compute predictions
    ypred = [model(xi) for xi in X]
    
    # Calculate SVM Hinge Loss (max(0, 1 - y_gt * y_pred))
    hinge_losses = [(1 + -ygt * yout).relu() for ygt, yout in zip(y, ypred)]
    data_loss = sum(hinge_losses) / len(X)
    
    # Regularization: penalize large weights (L2 regularization)
    reg_loss = regularization_strength * sum(p*p for p in model.parameters())
    total_loss = data_loss + reg_loss
    
    # Backpropagation
    model.zero_grad()
    total_loss.backward()
    
    # Gradient Descent / weight update (with learning rate decay)
    lr = learning_rate - 0.9 * epoch / epochs
    for p in model.parameters():
        p.data -= lr * p.grad
        
    # Calculate accuracy
    accuracy = sum((ygt > 0) == (yout.data > 0) for ygt, yout in zip(y, ypred)) / len(y)
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {total_loss.data:.4f} | Accuracy: {accuracy * 100:.1f}%")

# 4. Plot Decision Boundary
print("\nPlotting decision boundary...")
h = 0.25
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Classify grid points
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_preds = []
for gp in grid_points:
    pred = model(gp).data
    grid_preds.append(pred)
Z = np.array(grid_preds).reshape(xx.shape)

# Draw plot
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors='black')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Graddy MLP Decision Boundary on Moons Dataset")
output_image = "moons_decision_boundary.png"
plt.savefig(output_image)
print(f"Saved decision boundary plot to '{output_image}'.")
