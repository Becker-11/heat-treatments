import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Model Parameters
v0 = 10          # Initial condition coefficient
u0 = 1000              # Source term coefficient
D0 = 0.0138
E_a = 131e3
A_D = 0.9e9   
E_AD = 111.53e3
R = 8.314
T = 418.15    # 280C = 553.15K

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

def plot_oxygen_concentration(x_range, t_values):
    """
    Plots oxygen concentration vs distance for a given range of distances and multiple time values.
    
    Parameters:
        x_range (tuple): A tuple of the form (min_x, max_x) defining the range of distances (in nm).
        t_values (list): A list of time values at which to calculate the oxygen concentration (in seconds).
    """
    x_values = np.linspace(x_range[0], x_range[1], 1000)  # Distance range
    
    
    plt.figure(figsize=(10, 6))
    
    for t_value in t_values:
        concentration_values = [c(x, t_value) for x in x_values]
        plt.plot(x_values, concentration_values, label=f't={t_value/3600:.1f} h')
    
    # Set titles and labels with units
    plt.title('Oxygen Concentration vs Distance')
    plt.xlabel('Distance (x) [nm]')
    plt.ylabel('Oxygen Concentration (at. %)')
    
    # Set y-axis limits dynamically based on all concentration values
    plt.ylim(0, plt.ylim()[1] * 1.1)  # Adjust y-axis limits

    plt.legend()
    plt.grid()
    plt.show()


def main():
    plot_oxygen_concentration((0, 300), [45*3600])
    # print(D(T))
    # print(k(T))
    # print(v(0, 1))
    # print(quad(test_integrand, 0, 1, args=(1, 0)))  



if __name__ == "__main__":
    main()  
