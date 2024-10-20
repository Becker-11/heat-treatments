import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 5e-4               # Length of spatial domain (cm)
total_time = 3      # Total time (seconds)
N = 1000               # Number of spatial points
M = 1000              # Number of time steps
dx = L / N            # Spatial step size
dt = total_time / M        # Time step size

v0 = 3.5           # Initial condition coefficient
u0 = 200              # Source term coefficient
D0 = 0.075
E_a = 131e3
A_D = 0.9e9   
E_AD = 119.9e3
R = 8.314

T = 550    # Kelvin


# Discretized space and time
x = np.linspace(0, L, N)
t = np.linspace(0, total_time, M)

# Initial condition: Gaussian approximation of delta function
sigma = 0.01
c = np.zeros((M, N))
c[0, :] = v0 * np.exp(-x**2 / (2 * sigma**2))
       

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


def simulate_heat_treatment():
    # Time-stepping loop
    for j in range(0, M - 1):
        T_current = T
        for i in range(1, N - 1):   
            D_T = D(T_current)
            q_t = q_source(t[j], T_current)
            c[j+1, i] = c[j, i] + (D_T * dt / dx**2) * (c[j, i+1] - 2*c[j, i] + c[j, i-1]) + dt * q_t


def plot_solution():
    plt.plot(x, c, label=f"t={total_time}")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("c(x,t)")
    plt.show()


def main():
    simulate_heat_treatment()
    plot_solution()


if __name__ == "__main__":
    main()  
