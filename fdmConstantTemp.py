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


# Source term q(x, t)
def q(t):
    return u0 * k(T) * np.exp(-k(T)*t)


# Finite difference method for solving the PDE
def finite_difference_v(x_range, t_max, nx, nt):
    # Spatial and time step sizes
    x_min, x_max = x_range
    dx = (x_max - x_min) / (nx - 1)  # Spatial step size
    dt = t_max / nt  # Time step size
    
    # Discretize the spatial domain
    x_values = np.linspace(x_min, x_max, nx)
    
    # Initialize v(x, t) array
    v = np.zeros((nx, nt))
    # Set initial condition at t = 0
    #v[:, 0] = v0 * gaussian_delta(x_values)
    v[0,0] = v0

    # Time stepping loop
    for n in range(0, nt - 1):
        D_t = D(T)  # Assume constant temperature for now
        for i in range(1, nx - 1):  # Skip boundary points
            # Finite difference update rule for interior points
            v[i, n+1] = v[i, n] + (D_t * dt / dx**2) * (v[i+1, n] - 2 * v[i, n] + v[i-1, n])
        
        # Apply zero-flux boundary conditions
        v[0, n+1] = v[1, n+1]  # Left boundary
        v[-1, n+1] = v[-2, n+1]  # Right boundary
    
    return v, x_values

def finite_difference_u(x_range, t_max, nx, nt):
# Spatial and time step sizes
    x_min, x_max = x_range
    dx = (x_max - x_min) / (nx - 1)  # Spatial step size
    dt = t_max / nt  # Time step size
    
    # Discretize the spatial domain
    x_values = np.linspace(x_min, x_max, nx)

    # Initialize u(x, t) array
    u = np.zeros((nx, nt))
    # Initial condition at t = 0
    u[0, 0] = u0 * k(T) # Concentrate all initial oxygen at the first grid point (x = 0)

    # Time stepping loop
    for n in range(0, nt - 1):
        D_t = D(T)  # Assume constant temperature for now
        for i in range(1, nx - 1):  # Skip boundary points
            # Finite difference update rule for interior points
            u[i, n+1] = u[i, n] + (D_t * dt / dx**2) * (u[i+1, n] - 2 * u[i, n] + u[i-1, n]) + q(n*dt) * dt

        # Boundary condition at x = 0 (apply source term)
        u[0, n+1] = u[1,n] + (D_t * dt / dx**2) * (u[2, n] - 2 * u[1, n] + u[0, n]) + q(n*dt) * dt
        # Boundary condition at x = x_max (Dirichlet condition: u -> 0 as x -> infinity)
        u[-1, n+1] = 0  # Dirichlet condition at the right boundary
    
    return u, x_values


def analytic_v(x,t):
    return (v0 / np.sqrt(np.pi * D(T) * t)) * np.exp(-x**2 / (4 * D(T) * t))


def analytic_u(x,t):
    int_val, err = quad(u_integrand, 0, t, args=(t, x), epsabs=1e-10, epsrel=1e-10)
    return int_val

def u_integrand(s, t, x):
    return (u0 * k(T) * np.exp(-k(T) * t) / np.sqrt(np.pi * D(T) * (t - s))) * np.exp(-x**2 / (4 * D(T) * (t - s)))



# Plotting function for the solution
def plot_concentrations(v, u, x_values, t_values, dt):
    plt.figure(figsize=(10, 6))
    
    for t_value in t_values:
        t_index = int(t_value / dt)
        plt.plot(x_values, v[:, t_index-1], label=f'fdm v(x,t), t={t_value/3600:.1f} h')
        plt.plot(x_values, analytic_v(x_values, t_value), label=f'analytic v(x,t), t={t_value/3600:.1f} h')
        plt.plot(x_values, u[:, t_index-1], label=f'fdm u(x,t), t={t_value/3600:.1f} h')
        plt.plot(x_values, [analytic_u(x, t_value) for x in x_values], label=f'analytic u(x,t), t={t_value/3600:.1f} h')

    # Set titles and labels with unitsd
    plt.title('v(x,t) and u(x,t) vs Distance')
    plt.xlabel('Distance (x) [nm]')
    plt.ylabel('Concentration')
    plt.legend()
    plt.grid()
    plt.show()



def main():
    # Define simulation parameters
    x_range = (0, 300)  # Spatial domain in nm
    t_max = 45*3600  # Maximum time in seconds
    nx = 150 # Number of spatial points
    nt = 1297 # Number of time steps (200, 69082 for v(x,t) w/ v[:,0] = v0 * gaussian_delta(x_values))
    t_values = [45*3600]  # Plot at 45h

    # Run the finite difference method simulation
    v, x_values = finite_difference_v(x_range, t_max, nx, nt)  
    u, x_values = finite_difference_u(x_range, t_max, nx, nt)
    # Plot the v(x,t) and u(x,t) profiles
    plot_concentrations(v, u, x_values, t_values, t_max / nt)

    print(v[0,nt-1])
    print(analytic_v(0, t_max))
    print(u[0,nt-1])
    print(analytic_u(0, t_max))

if __name__ == "__main__":
    main()

