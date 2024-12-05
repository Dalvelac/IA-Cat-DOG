import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# load weight values from file (assuming you save weights periodically during training)
weights = np.loadtxt("weights.csv", delimiter=",")

# apply T-SNE to reduce dimensions
tsne = TSNE(n_components=2, random_state=42)
reduced_weights = tsne.fit_transform(weights)

# plot the trajectories
plt.figure(figsize=(10, 8))
for i in range(5):  # Assuming 5 different initializations
    plt.plot(reduced_weights[i::5, 0], reduced_weights[i::5, 1], marker='o', label=f'Trajectory {i+1}')
plt.title("Optimization Trajectories with T-SNE")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.show()
