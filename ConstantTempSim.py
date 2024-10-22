import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Model Parameters
v0 = 10          # Initial condition coefficient
u0 = 1000              # Source term coefficient
D0 = 0.0138
E_a = 135e3
A_D = 3e9   
E_AD = 111.53e3
R = 8.314
T = 418.15    # 280C = 553.15K

# Gaussian approximation for Dirac delta function at x = 0
def gaussian_delta(x, sigma=0.01):
    return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

def D(T):
    nmpercm = 1e7
    return nmpercm **2 * D0 * np.exp(-E_AD / (R * T)) 

def k(T):
    return A_D * np.exp(-E_a / (R * T))

def v(x,t):
    return (v0 / np.sqrt(np.pi * D(T) * t)) * np.exp(-x**2 / (4 * D(T) * t))

def u(x,t):
    int_val, err = quad(u_integrand, 0, t, args=(t, x), epsabs=1e-10, epsrel=1e-10)
    return int_val

def u_integrand(s, t, x):
    return (u0 * k(T) * np.exp(-k(T) * t) / np.sqrt(np.pi * D(T) * (t - s))) * np.exp(-x**2 / (4 * D(T) * (t - s)))

def c(x, t):
    return u(x,t) + v(x,t)


def plot_oxygen_concentration(ax, x_range, t_values):
    """
    Plots oxygen concentration vs distance for a given range of distances and multiple time values.
    
    Parameters:
        ax: The axis to plot on (for subplots).
        x_range (tuple): A tuple of the form (min_x, max_x) defining the range of distances (in nm).
        t_values (list): A list of time values at which to calculate the oxygen concentration (in seconds).
    """
    x_values = np.linspace(x_range[0], x_range[1], 1000)  # Distance range
    
    for t_value in t_values:
        concentration_values = [c(x, t_value) for x in x_values]
        ax.plot(x_values, concentration_values, label=f't={t_value / 3600:.1f} h')
    
    # Set titles and labels with units
    ax.set_title('Oxygen Concentration vs Distance')
    ax.set_xlabel('Distance (nm)')
    ax.set_ylabel('Oxygen Concentration (at. %)')
    ax.legend()
    ax.grid()

def plot_colormesh_and_concentration(x_range, t_range, t_values):
    """
    Generates two subplots: one with a pcolormesh plot for oxygen concentration as a function of time and distance,
    and another with oxygen concentration vs distance for specific times.
    
    Parameters:
        x_range (tuple): A tuple of the form (min_x, max_x) defining the range of distances (in nm).
        t_range (tuple): A tuple of the form (min_t, max_t) defining the range of time values (in seconds).
        t_values (list): A list of time values at which to calculate the oxygen concentration (in seconds).
    """
    # Create figure with two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # First subplot: pcolormesh plot
    x_values = np.linspace(x_range[0], x_range[1], 100)  # Distance range
    t_values_colormesh = np.linspace(t_range[0], t_range[1], 100)  # Time range in seconds
    X, T = np.meshgrid(x_values, t_values_colormesh)
    
    concentration_matrix = np.vectorize(c)(X, T)

    mesh = axs[0].pcolormesh(X, T / 3600, concentration_matrix, shading='auto', cmap='jet', vmin=0, vmax=0.3)  # Colorbar limits from 0 to 0.3
    fig.colorbar(mesh, ax=axs[0], label='Oxygen Concentration (at. %)')

    # Set titles and labels for the colormesh plot
    axs[0].set_title('Oxygen Concentration as a Function of Distance and Time')
    axs[0].set_xlabel('Distance (nm)')
    axs[0].set_ylabel('Time (hours)')

    # Second subplot: Oxygen concentration vs distance
    plot_oxygen_concentration(axs[1], x_range, t_values)  # Plot on the second axis

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the combined plot
    plt.show()

def main():
    x_range = (0, 300)  # Distance range in nm
    t_range = (0, 45 * 3600)  # Time range in seconds (up to 45 hours)
    t_values = [5 * 3600, 15 * 3600, 30 * 3600, 45 * 3600]  # Time values to plot in seconds
    
    # Plot both the colormesh and the concentration vs distance plots
    plot_colormesh_and_concentration(x_range, t_range, t_values)

if __name__ == "__main__":
    main()