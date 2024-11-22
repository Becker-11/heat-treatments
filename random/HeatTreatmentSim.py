import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Model Parameters
v0 = 10          # Initial condition coefficient
u0 = 1000        # Source term coefficient
D0 = 0.0138      # Diffusion constant
E_a = 131e3      # Activation energy for k(T)
A_D = 0.9e9      # Pre-exponential factor for k(T)
E_AD = 111.53e3  # Activation energy for D(T)
R = 8.314        # Gas constant

# Time when temperature changes phases (in seconds)
time_increase_start = 13392  # Time increase starts
time_decrease_start = 30348  # Time decrease starts

# Temperature function T(t)
def T(t):
    if t <= time_increase_start:
        return -0.97 * t + 306.92
    elif time_increase_start < t <= time_decrease_start:
        return 62.29 * t + 57.02
    else:
        return 1117.50 * np.exp(-0.17 * t) + 307.66

# Diffusion coefficient D(T)
def D(T):
    nmpercm = 1e7
    return nmpercm ** 2 * D0 * np.exp(-E_AD / (R * T))

# Rate constant k(T)
def k(T):
    return A_D * np.exp(-E_a / (R * T))

# Gaussian approximation for Dirac delta function at x = 0
def gaussian_delta(x, sigma=0.01):
    return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

# Source term q(t, x)
def q(t, x, sigma=0.01):
    # Integrate k(T) from 0 to t for the source term
    int_val, err = quad(lambda s: k(T(s)), 0, t, epsabs=1e-10, epsrel=1e-10)
    
    # Gaussian approximation for Dirac delta at x = 0
    delta_x = gaussian_delta(x, sigma)
    
    # Source term with Gaussian Dirac delta approximation
    return u0 * k(T(t)) * np.exp(-int_val) * delta_x

# Finite difference method to solve Fick's second law
def finite_difference_ficks_law(D, q, x_max, t_max, nx, nt, sigma=0.01):
    dx = x_max / (nx - 1)  # Spatial step size
    dt = t_max / nt        # Time step size

    # Initialize concentration array c(x, t)
    c = np.zeros((nx, nt))  
    x = np.linspace(0, x_max, nx)
    
    # Initial condition: Concentration at x = 0
    c[0, 0] = v0
    
    # Time stepping loop
    for n in range(0, nt-1):
        T_t = T(n * dt)  # Calculate temperature at current time
        D_t = D(T_t)     # Diffusion coefficient at current time
        
        for i in range(1, nx-1):  # Interior points only
            q_t = q(n * dt, x[i], sigma)  # Source term at x[i] and current time
            
            # Finite difference update
            c[i, n+1] = c[i, n] + (D_t * dt / dx**2) * (c[i+1, n] - 2 * c[i, n] + c[i-1, n]) + dt * q_t

        # Boundary conditions (zero flux at both ends)
        c[0, n+1] = c[1, n+1]
        c[-1, n+1] = c[-2, n+1]

    return c, x

# Function to plot the concentration profile at a given time
def plot_concentration_profile(c, x, t_max):
    plt.figure(figsize=(10, 6))
    plt.plot(x, c[:, -1], label=f't={t_max:.2f} s')
    plt.xlabel('Position x (nm)')
    plt.ylabel('Concentration c(x,t)')
    plt.legend()
    plt.title('Concentration Profile at Final Time')
    plt.grid(True)
    plt.show()

# Main function to run the simulation
def main():
    # Spatial and time parameters
    x_max = 4000  # Maximum spatial domain (in nm)
    t_max = 45*3600  # Maximum time (in seconds)
    nx = 100  # Number of spatial points
    nt = 2000  # Number of time steps
    sigma = 10  # Width of Gaussian for Dirac delta approximation (in nm)

    # Solve Fick's second law using the finite difference method
    c, x = finite_difference_ficks_law(D, q, x_max, t_max, nx, nt, sigma)

    # Plot the final concentration profile
    plot_concentration_profile(c, x, t_max)

if __name__ == "__main__":
    main()
