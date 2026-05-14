import numpy as np

# McCulloch-Pitts Neural Network for ANDNOT Function
# Output = A AND (NOT B)

# Step activation function
def activation(x):
    if x >= 1:
        return 1
    else:
        return 0

# Input combinations
inputs = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

# Weights
w1 = 1      # weight for A
w2 = -1     # inhibitory weight for B
threshold = 1

print("A B  Output")

# Processing inputs
for x in inputs:
    net = x[0]*w1 + x[1]*w2
    y = activation(net)
    print(x[0], x[1], "   ", y)