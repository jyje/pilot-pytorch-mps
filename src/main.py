import torch
import math

## Configuration
# Set data type
dtype = torch.float

# Set device: Check for available devices and select one
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("MPS is available and built. Using Apple Silicon GPU.")
else:
    device = torch.device("cpu")
    print("Neither CUDA nor MPS is available or built. Using CPU.")


# Generate training data (linear data)
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype).unsqueeze(1)
y = 3 * x + torch.randn(x.size(), device=device, dtype=dtype)  # y = 3x + noise

# Initialize model (random weights)
w = torch.randn((1, 1), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-3
for t in range(5000):
    # Compute predictions
    y_pred = x.mm(w) + b

    # Compute loss (mean squared error)
    loss = (y_pred - y).pow(2).mean().item()
    if t % 100 == 0:
        print(t, loss)

    # Compute gradients via backpropagation
    grad_y_pred = 2.0 * (y_pred - y) / x.size(0)
    grad_w = x.t().mm(grad_y_pred)
    grad_b = grad_y_pred.sum()

    # Update weights
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

print(f'Final loss: {loss}')
print(f'Result: y = {w.item()}x + {b.item()}')
