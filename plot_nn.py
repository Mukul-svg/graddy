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
y = y * 2 - 1

# 2. Define a clean, readable MLP size [2, 8, 8, 1]
model = MLP(2, [8, 8, 1])
print(f"Initialized MLP model for plotting with {len(model.parameters())} parameters.")

# 3. Quick training loop to fit the data
epochs = 150
learning_rate = 0.5
regularization_strength = 1e-4

print("Training MLP model...")
for epoch in range(epochs):
    ypred = [model(xi) for xi in X]
    hinge_losses = [(1 + -ygt * yout).relu() for ygt, yout in zip(y, ypred)]
    data_loss = sum(hinge_losses) / len(X)
    reg_loss = regularization_strength * sum(p*p for p in model.parameters())
    total_loss = data_loss + reg_loss
    
    model.zero_grad()
    total_loss.backward()
    
    lr = learning_rate - 0.4 * epoch / epochs
    for p in model.parameters():
        p.data -= lr * p.grad

print(f"Final training loss: {total_loss.data:.4f}")

# 4. Neural Network Plotting Function
def draw_neural_network(model, filename="nn_architecture.png"):
    # Extract layer sizes
    input_size = len(model.layers[0].neurons[0].w)
    layer_sizes = [input_size] + [len(layer.neurons) for layer in model.layers]
    
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.axis('off')
    
    # Layer spacing
    x_space = 2.0
    layer_coords = []
    
    # Compute node coordinates
    for i, size in enumerate(layer_sizes):
        x = i * x_space
        if size > 1:
            y_vals = np.linspace(-size/2.0, size/2.0, size)
        else:
            y_vals = np.array([0.0])
        coords = [(x, y) for y in y_vals]
        layer_coords.append(coords)
        
    # Draw connections (weights) first
    for i in range(len(model.layers)):
        layer = model.layers[i]
        prev_coords = layer_coords[i]
        curr_coords = layer_coords[i+1]
        
        # Max weight absolute value for normalization
        all_weights = [abs(w.data) for n in layer.neurons for w in n.w]
        max_w = max(all_weights) if all_weights else 1.0
        
        for j, neuron in enumerate(layer.neurons):
            dest_coord = curr_coords[j]
            for k, w_val in enumerate(neuron.w):
                src_coord = prev_coords[k]
                w = w_val.data
                
                # Green/blue color for positive weights, red/orange for negative weights
                color = '#2E7D32' if w > 0 else '#C62828'
                alpha = max(0.1, min(0.85, abs(w) / max_w))
                linewidth = 0.5 + 2.5 * (abs(w) / max_w)
                
                ax.plot([src_coord[0], dest_coord[0]], [src_coord[1], dest_coord[1]], 
                        color=color, alpha=alpha, linewidth=linewidth, zorder=1)
                
    # Draw nodes and annotate
    colors = ['#1976D2', '#7B1FA2', '#7B1FA2', '#E65100'] # Input, Hidden 1, Hidden 2, Output
    labels = ['Input Layer', 'Hidden Layer 1', 'Hidden Layer 2', 'Output Layer']
    
    for i, size in enumerate(layer_sizes):
        coords = layer_coords[i]
        color = colors[min(i, len(colors)-1)]
        label_name = labels[min(i, len(labels)-1)]
        
        for j, coord in enumerate(coords):
            # Draw circle
            circle = plt.Circle(coord, radius=0.12, color=color, zorder=2, ec='black', lw=1.2)
            ax.add_patch(circle)
            
            # Label node index inside the circle
            ax.text(coord[0], coord[1], f"{j}", color='white', ha='center', va='center', 
                    fontsize=8, fontweight='bold', zorder=3)
            
            # Show bias value if it's a neuron (not input layer)
            if i > 0:
                neuron = model.layers[i-1].neurons[j]
                bias_val = neuron.b.data
                ax.text(coord[0], coord[1] - 0.22, f"b={bias_val:.2f}", color='#333333', 
                        ha='center', va='top', fontsize=6, fontweight='bold', zorder=3)
                
        # Draw layer label at the top
        top_y = max([c[1] for c in coords]) + 0.4
        ax.text(coords[0][0], top_y, f"{label_name}\n({size} Neurons)" if i > 0 else f"{label_name}\n({size} Inputs)", 
                ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)
        
    plt.title("Architecture Visualization (Weights & Biases)", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved neural network diagram to '{filename}'.")

# Execute
draw_neural_network(model)
