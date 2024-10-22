import numpy as np
import matplotlib.pyplot as plt

# Original Gaussian approximation for Dirac delta
def gaussian_delta(x, sigma=0.01):
    return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

# Flat-top Gaussian (Super-Gaussian) approximation
def flat_top_gaussian(x, sigma=0.01, n=4):
    """
    Flat-top Gaussian (super-Gaussian) to approximate Dirac delta function.
    
    Parameters:
        x (float or np.array): Input value(s) where the function is evaluated.
        sigma (float): Standard deviation or width of the Gaussian.
        n (int): Exponent controlling the flatness of the top.
    
    Returns:
        float or np.array: Flat-top Gaussian values.
    """
    A = 1 / (sigma * np.sqrt(2 * np.pi))  # Normalization factor
    return A * np.exp(-((x / sigma) ** (2 * n)))

# Define the range of x-values
x_values = np.linspace(-0.05, 0.05, 1000)

# Parameters for both approximations
sigma = 0.001
n = 10  # Exponent for flat-top Gaussian

# Calculate the values of both functions
gaussian_values = gaussian_delta(x_values, sigma=sigma)
flat_top_values = flat_top_gaussian(x_values, sigma=sigma, n=n)

# Plot both functions for comparison
plt.figure(figsize=(10, 6))
plt.plot(x_values, gaussian_values, label='Gaussian Delta Approximation', color='blue')
plt.plot(x_values, flat_top_values, label=f'Flat-Top Gaussian (n={n})', color='red', linestyle='--')

plt.title('Comparison of Gaussian and Flat-Top Gaussian Dirac Delta Approximations')
plt.xlabel('x')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
