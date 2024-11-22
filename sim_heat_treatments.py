import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from crank_nicolson_fdm import CrankNicolsonFdm
from temperature_generator import TemperatureGenerator



# Model parameters (example values)
v0 = 10
u0 = 1000
D0 = 0.0138
E_a = 131e3
A_D = 0.9e9   
E_AD = 111.53e3
R = 8.314
#T = 418.15    # Temperature in Kelvin

def gaussian_delta(x, sigma=0.001, n=6):
    A = 1 / (sigma * np.sqrt(2 * np.pi))  
    return A * np.exp(-((x / sigma) ** (2 * n)))

def D(T):
    nmpercm = 1e7
    return nmpercm **2 * D0 * np.exp(-E_AD / (R * T))

def k(T):
    return A_D * np.exp(-E_a / (R * T))

def q(x, t):
    return u0 * k(T) * np.exp(-k(T)*t) * gaussian_delta(x)

def analytic_v(x,t):
    return (v0 / np.sqrt(np.pi * D(T) * t)) * np.exp(-x**2 / (4 * D(T) * t))

def analytic_u(x,t):
    int_val, err = quad(u_integrand, 0, t, args=(t, x), epsabs=1e-10, epsrel=1e-10)
    return int_val

def u_integrand(s, t, x):
    return (u0 * k(T) * np.exp(-k(T) * t) / np.sqrt(np.pi * D(T) * (t - s))) * np.exp(-x**2 / (4 * D(T) * (t - s)))

def analytic_c(x, t):
    return analytic_u(x,t) + analytic_v(x, t)  

# Plotting function for the solutions
def plot_concentrations(C, x_grid, tf):
    plt.figure(figsize=(10, 6))

    a_u = [analytic_u(x, tf) for x in x_grid]
    a_v = analytic_v(x_grid, tf)
    a_c = a_u + a_v

    print(C[0])
    print(a_c[0])

    # Plot for v(x,t)
    plt.plot(x_grid, a_v, label=f'analytic v(x,t), t={tf/3600:.1f} h')
    # Plot for u(x,t)
    plt.plot(x_grid, a_u, label=f'analytic u(x,t), t={tf/3600:.1f} h')
    # Combine u(x,t) and v(x,t) for c(x,t)
    plt.plot(x_grid, a_c, label=f'analytic c(x,t), t={tf/3600:.1f} h')
    plt.plot(x_grid, C, alpha=0.7, linestyle='--', label=f'fdm c(x,t), t={tf/3600:.1f} h')

    # Set titles and labels with unitsd
    plt.title('c(x,t), v(x,t) and u(x,t) vs Distance')
    plt.xlabel('Distance (x) [nm]')
    plt.ylabel('Concentration % (nm)')
    plt.legend()
    plt.grid()
    plt.show()


def main():

    # Define parameters and functions
    L = 300.0     # Spatial domain in nm
    nx = 830      # Number of spatial points
    tf = 45*3600  # Final time in seconds
    nt = 6000     # Number of time steps

    T = TemperatureGenerator("heating_instructions.yaml")
    tf = T.t_max * 3600

    # Create an instance of the solver
    fdm_solver = CrankNicolsonFdm(L=L, nx=nx, tf=tf, nt=nt, D=D, q=q, v0=v0, T=T)

    # Run the solver
    fdm_solver.crank_nicolson()

    # Retrieve the solution and grids
    C_record = fdm_solver.get_solution()
    x_grid, t_grid = fdm_solver.get_grid()

    plot_concentrations(C_record[-1,:], x_grid, tf)


if __name__ == "__main__":
    main()


