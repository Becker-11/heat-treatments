import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0               # Length of spatial domain
total_time = 0.1      # Total time
N = 100               # Number of spatial points
M = 1000              # Number of time steps
dx = L / N            # Spatial step size
dt = total_time / M        # Time step size

v0 = 3.5           # Initial condition coefficient
u0 = 200              # Source term coefficient
D0 = 0.075
E_a = 131
A_D = 0.9
E_AD = 119.9
R = 8.314

T = 280
       

def delta_gaussian(x, sigma):
    return np.exp(-x**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)


def D(T):
    return D0 * np.exp(-E_a / (R * T))


def k(T):
    return A_D * np.exp(-E_AD / (R * T))

def q_source(t, T):
    x = np.linspace(0, L, N)
    sigma = 0.01
    return u0 * k(T) * np.exp(-k(T) * t) #* delta_gaussian(x, sigma)

print(q_source(0, T))

# Discretized space and time
x = np.linspace(0, L, N)
t = np.linspace(0, total_time, M)

# Initial condition: Gaussian approximation of delta function
sigma = 0.01
c = np.zeros((M, N))
c[0, :] = v0 * np.exp(-x**2 / (2 * sigma**2))

# Time-stepping loop
for j in range(0, M - 1):
    T_current = T
    for i in range(1, N - 1):   
        D_T = D(T_current)
        q_t = q_source(t[j], T_current)
        c[j+1, i] = c[j, i] + (D_T * dt / dx**2) * (c[j, i+1] - 2*c[j, i] + c[j, i-1]) + dt * q_t

# Plot the solution at different times
plt.plot(x, c[0, :], label="t=0")
plt.plot(x, c[M//2, :], label=f"t={total_time/2}")
plt.plot(x, c[-1, :], label=f"t={total_time}")
plt.legend()
plt.xlabel("x")
plt.ylabel("c(x,t)")
plt.show()
