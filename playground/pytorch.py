import torch
import numpy as np

# Create a 1D tensor
x = torch.tensor([1.0, 2.0, 3.0])
# Move tensor to GPU if available
x = x.to("cuda") if torch.cuda.is_available() else x
print(x)

# Create a 1D numpy array
y = np.array([1.0, 2.0, 3.0])
print(y)
