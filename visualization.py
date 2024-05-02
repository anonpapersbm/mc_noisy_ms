import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib import pyplot as plt
import numpy as np

n = 5
epsilon = 1e-4

z = np.zeros(n)
for k in range(int(np.ceil(n/2))):
    z[2*k] = 1
z_tilde = z + np.random.normal(0, epsilon, n)
M_star = np.outer(z, z)
M_dagger = np.outer(z_tilde, z_tilde)

# Create a reshaped and squared version of M_star and M_dagger-M_star once
M_star_reshaped = np.reshape(M_star, -1)**2
M_dagger_diff = np.reshape(M_dagger - M_star, -1)**2

for p in [0.2,0.3, 0.4, 0.5,0.6]:
    epsilons = np.linspace(1e-5, 1e-2, 100)
    distances = np.linspace(1e-3, 0.1, 100)
    xx, yy = np.meshgrid(distances, epsilons)

    # Calculate etas based on distances and epsilons
    # Ratios should be computed with respect to the meshgrid of distances and epsilons
    ratios = xx / (np.linalg.norm(M_star)**2 * yy**2) 
    etas = p - 1 / (1 + ratios)

    # Calculate d for each combination of epsilons and etas using broadcasting
    d = M_dagger_diff + (yy**2)[:, :, np.newaxis] * M_star_reshaped

    # Calculate probabilities
    norm_d1 = np.sum(d, axis=2)  # axis=2 to sum over the flattened matrix dimensions
    norm_d2 = np.sqrt(np.sum(d**2, axis=2))
    probs = 1 - np.exp(-2 * etas**2 * norm_d1**2 / norm_d2**2)

    print("Distances range:", np.min(distances), "to", np.max(distances))
    print("Probability range:", np.min(probs), "to", np.max(probs))

    plt.figure(figsize=(10, 7))
    # Plotting using distances as x-axis and epsilons as y-axis
    contour = plt.contourf(xx, yy, probs, levels=30, cmap='magma')
    plt.colorbar(contour)
    plt.xlabel(r'Upper-Bound on $\|M^\dagger - M^*\|^2_F$', size=30)
    plt.ylabel('Epsilon Value', size=25)
    plt.title(f'Probability Contour Plot for p = {p}')
    plt.show()