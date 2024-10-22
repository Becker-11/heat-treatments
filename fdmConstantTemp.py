import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import time

# Model Parameters
v0 = 10          # Initial condition coefficient
u0 = 1000              # Source term coefficient
D0 = 0.0138
E_a = 131e3
A_D = 0.9e9   
E_AD = 111.53e3
R = 8.314
T = 418.15    # 280C = 553.15K

# Gaussian approximation for Dirac delta function at x = 0
#def gaussian_delta(x, sigma=0.01):
#    return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

# Flat-top Gaussian (Super-Gaussian) approximation (300, x, 0.00012)
def gaussian_delta(x, sigma=0.00689, n=6):
    A = 1 / (sigma * np.sqrt(2 * np.pi))  
    return A * np.exp(-((x / sigma) ** (2 * n)))


def D(T):
    nmpercm = 1e7
    return nmpercm **2 * D0 * np.exp(-E_AD / (R * T)) 

def k(T):
    return A_D * np.exp(-E_a / (R * T))

# Source term q(x, t)
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

def fdm_u_v(x_range, t_max, nx, nt):
    """
    Finite difference method for solving both u(x,t) and v(x,t) PDEs simultaneously.

    Parameters:
    - x_range: tuple (x_min, x_max), the range of the spatial domain.
    - t_max: float, maximum time.
    - nx: int, number of spatial grid points.
    - nt: int, number of time steps.
    
    Returns:
    - v: the solution array for v(x,t).
    - u: the solution array for u(x,t).
    - x_values: array of spatial points.
    """
    
    # Spatial and time step sizes
    x_min, x_max = x_range
    dx = (x_max - x_min) / (nx - 1)  # Spatial step size
    dt = t_max / nt  # Time step size
    
    # Discretize the spatial domain
    x_values = np.linspace(x_min, x_max, nx)
    t_values = np.linspace(0, t_max, nt)
    # Initialize v(x, t) and u(x, t) arrays
    v = np.zeros((nx, nt))
    u = np.zeros((nx, nt))
    
    # Set initial conditions at t = 0
    v[:, 0] = v0 * gaussian_delta(x_values)
    u[0, :] = u0 * k(T) * np.exp(-k(T) * t_values)

    #u[0, :] = q(0, t_values * dt)
    #u[:, 0] = u0 * k(T) * gaussian_delta(0)

    # Precompute constant to avoid recalculating each step
    coeff_v = D(T) * dt / dx**2
    coeff_u = D(T) * dt / dx**2

    # Time-stepping loop (vectorized)
    for n in range(0, nt - 1):
        # Vectorized update for all interior points (no loop needed)
        v[1:-1, n+1] = v[1:-1, n] + coeff_v * (v[2:, n] - 2 * v[1:-1, n] + v[:-2, n])
        #u[1:-1, n+1] = u[1:-1, n] + coeff_u * (u[2:, n] - 2 * u[1:-1, n] + u[:-2, n]) + q(x_values[1:-1], n*dt) * dt
        u[1:-1, n+1] = u[1:-1, n] + coeff_u * (u[2:, n] - 2 * u[1:-1, n] + u[:-2, n]) + q(x_values[1:-1], n) * dt


        # Apply boundary conditions at x = 0 (source term for u)
        v[0, n+1] = v[1, n+1]  # Zero-flux boundary for v
        #u[0, n+1] = u[0, n+1] + q(x_values[0], (n+1) * dt) * dt
        u[0, n+1] = u[0, n] + coeff_u * (u[1, n] - 2 * u[0, n] + u[-1, n]) + q(x_values[0], n*dt) * dt

        # Boundary condition at x = x_max (Dirichlet condition: u -> 0 as x -> infinity)
        #v[-1, n+1] = v[-2, n+1]  # Zero-flux boundary for v
        #u[-1, n+1] = 0  # Dirichlet condition for u at the right boundary
    
    return u, v, x_values


# Plotting function for the solutions
def plot_concentrations(u, v, x_values, t_values, dt):
    plt.figure(figsize=(10, 6))
    
    for t_value in t_values:
        t_index = int(t_value / dt)

        a_u = [analytic_u(x, t_value) for x in x_values]
        a_v = analytic_v(x_values, t_value)
        a_c = a_u + a_v

        # Plot for v(x,t)
        plt.plot(x_values, a_v, label=f'analytic v(x,t), t={t_value/3600:.1f} h')
        plt.plot(x_values, v[:, t_index-1], alpha=0.7, linestyle='--', label=f'fdm v(x,t), t={t_value/3600:.1f} h')

        # Plot for u(x,t)
        plt.plot(x_values, a_u, label=f'analytic u(x,t), t={t_value/3600:.1f} h')
        plt.plot(x_values, u[:, t_index-1], alpha=0.7, linestyle='--', label=f'fdm u(x,t), t={t_value/3600:.1f} h')

        # Combine u(x,t) and v(x,t) for c(x,t)
        plt.plot(x_values, a_c, label=f'analytic c(x,t), t={t_value/3600:.1f} h')
        plt.plot(x_values, u[:, t_index-1] + v[:, t_index-1], alpha=0.7, linestyle='--', label=f'fdm c(x,t), t={t_value/3600:.1f} h')

    # Set titles and labels with unitsd
    plt.title('c(x,t), v(x,t) and u(x,t) vs Distance')
    plt.xlabel('Distance (x) [nm]')
    plt.ylabel('Concentration % (nm)')
    plt.legend()
    plt.grid()
    plt.show()

def test_u(u, x_values, t):
    plt.figure(figsize=(10, 6))

    a_u = [analytic_u(x, t_value) for x in x_values]

    # Plot for u(x,t)
    plt.plot(x_values, a_u, label=f'analytic u(x,t), t={t/3600:.1f} h')
    plt.plot(x_values, u[:, t_index-1], alpha=0.7, linestyle='--', label=f'fdm u(x,t), t={t/3600:.1f} h')


    # Set titles and labels with unitsd
    plt.title('u(x,t) vs Distance')
    plt.xlabel('Distance (x) [nm]')
    plt.ylabel('Concentration % (nm)')
    plt.legend()
    plt.grid()
    plt.show()



def main():

    start_time = time.time()

    # Define simulation parameters
    x_range = (0, 300)  # Spatial domain in nm
    t_max = 45*3600  # Maximum time in seconds
    nx = 300 # Number of spatial points (111, 7600, sigma=0.05) (300, 150000, 0.00689)
    nt = 150000 # Number of time steps (249, 86000, sigma=0.01) (556, 960000, sigma=0.002)
    dt = t_max / nt
    t_values = [45*3600]  # Plot at 45h


    # more space steps raises v(x,t) lowers u(x,t) capped? yes the time steps control the max value of u(x,t) at zero because it is dependent on
    # the approximation of the dirac delta which depends on sigma
    # decreasing sigma increases v(x,t) and u(x,t) at x = 0

    # Run the finite difference method simulation
    #v, x_values = finite_difference_v(x_range, t_max, nx, nt) 
    #u, x_values = finite_difference_u(x_range, t_max, nx, nt)
    u, v, x_values = fdm_u_v(x_range, t_max, nx, nt)
 
    # Plot the v(x,t) and u(x,t) profiles
    plot_concentrations(u, v, x_values, t_values, dt)

    print(f'v(0,nt-1):  {v[0,nt-1]}')
    print(f'v(0,t_max): {analytic_v(0, t_max)}')
    print(f'u(0,nt-1):  {u[0,nt-1]}')
    print(f'u(0,t_max): {analytic_u(0, t_max)}')

    print(f'\nTesting u(x,t) and fdm u(x,t)')
    print(f'fdm u(50, nt-1): {u[int(nx/6), nt-1]}')
    print(f'u(50, t_max):    {analytic_u(50,t_max)}')
    print(f'diff @ x=0:    {abs(analytic_u(0, t_max) - u[0, nt-1])}')
    print(f'diff @ x=50nm: {abs(analytic_u(50,t_max) - u[int(nx/6), nt-1])}')
    #print(abs(analytic_u(0, t_max) - u[0, nt-1]) + abs(analytic_u(50,t_max) - u[50, nt-1]))

    end_time = time.time()

    # Calculate and print execution time
    execution_time = end_time - start_time
    print(f"Simulation completed in {execution_time:.2f} seconds.")

if __name__ == "__main__":
    main()




# Finite difference method for solving the PDE
# def finite_difference_v(x_range, t_max, nx, nt):
#     # Spatial and time step sizes
#     x_min, x_max = x_range
#     dx = (x_max - x_min) / (nx - 1)  # Spatial step size
#     dt = t_max / nt  # Time step size
    
#     # Discretize the spatial domain
#     x_values = np.linspace(x_min, x_max, nx)
    
#     # Initialize v(x, t) array
#     v = np.zeros((nx, nt))
#     # Set initial condition at t = 0
#     v[:, 0] = v0 * gaussian_delta(x_values)
#     #v[0,0] = v0

#     # Time stepping loop
#     for n in range(0, nt - 1):
#         D_t = D(T)  # Assume constant temperature for now
#         for i in range(1, nx - 1):  # Skip boundary points
#             # Finite difference update rule for interior points
#             v[i, n+1] = v[i, n] + (D_t * dt / dx**2) * (v[i+1, n] - 2 * v[i, n] + v[i-1, n])
        
#         # Apply zero-flux boundary conditions
#         v[0, n+1] = v[1, n+1]  # Left boundary
#         v[-1, n+1] = v[-2, n+1]  # Right boundary
    
#     return v, x_values




# def finite_difference_u(x_range, t_max, nx, nt):
# # Spatial and time step sizes
#     x_min, x_max = x_range
#     dx = (x_max - x_min) / (nx - 1)  # Spatial step size
#     dt = t_max / nt  # Time step size
    
#     # Discretize the spatial domain
#     x_values = np.linspace(x_min, x_max, nx)

#     # Initialize u(x, t) array
#     u = np.zeros((nx, nt))
#     # Initial condition at t = 0
#     u[:,0] = u0 * k(T) * gaussian_delta(x_values)

#     # Time stepping loop
#     for n in range(0, nt - 1):
#         D_t = D(T)  # Assume constant temperature for now
#         for i in range(1, nx - 1):  # Skip boundary points
#             # Finite difference update rule for interior points
#             u[i, n+1] = u[i, n] + (D_t * dt / dx**2) * (u[i+1, n] - 2 * u[i, n] + u[i-1, n]) + q(x_values[i], n*dt) * dt

#         # Boundary condition at x = 0 (apply source term)
#         #u[0, n+1] = u[1,n] + (D_t * dt / dx**2) * (u[2, n] - 2 * u[1, n] + u[0, n]) + q(n*dt) * dt
#         #u[0, n+1] = u[0,n+1] + (D_t * dt / dx**2) * (u[1, n+1] - 2 * u[0, n+1] + u[-1, n+1]) + q(x_values[0], (n+1)*dt) * dt
#         u[0, n+1] = u[0,n] + (D_t * dt / dx**2) * (u[1, n] - 2 * u[0, n] + u[-1, n]) + q(x_values[0], (n)*dt) * dt

#         # Boundary condition at x = x_max (Dirichlet condition: u -> 0 as x -> infinity)
#         u[-1, n+1] = 0  # Dirichlet condition at the right boundary

#     return u, x_values






# def finite_difference_v(x_range, t_max, nx, nt):
#     # Spatial and time step sizes
#     x_min, x_max = x_range
#     dx = (x_max - x_min) / (nx - 1)  # Spatial step size
#     dt = t_max / nt  # Time step size
    
#     # Discretize the spatial domain
#     x_values = np.linspace(x_min, x_max, nx)
    
#     # Initialize v(x, t) array
#     v = np.zeros((nx, nt))
    
#     # Set initial condition at t = 0
#     v[:, 0] = v0 * gaussian_delta(x_values)

#     # Pre-compute constants to avoid repeating calculations
#     coeff = D(T) * dt / dx**2
    
#     # Time stepping loop (vectorized)
#     for n in range(0, nt - 1):
#         # Vectorized update for all interior points (no loop needed)
#         v[1:-1, n+1] = v[1:-1, n] + coeff * (v[2:, n] - 2 * v[1:-1, n] + v[:-2, n])
        
#         # Apply zero-flux boundary conditions
#         v[0, n+1] = v[1, n+1]  # Left boundary
#         v[-1, n+1] = v[-2, n+1]  # Right boundary
    
#     return v, x_values


# def finite_difference_u(x_range, t_max, nx, nt):
#     # Spatial and time step sizes
#     x_min, x_max = x_range
#     dx = (x_max - x_min) / (nx - 1)  # Spatial step size
#     dt = t_max / nt  # Time step size
    
#     # Discretize the spatial domain
#     x_values = np.linspace(x_min, x_max, nx)

#     # Initialize u(x, t) array
#     u = np.zeros((nx, nt))
    
#     # Initial condition at t = 0
#     u[:, 0] = u0 * k(T) * gaussian_delta(x_values)

#     # Precompute constant to avoid recalculating each step
#     coeff = D(T) * dt / dx**2

#     # Time-stepping loop (vectorized)
#     for n in range(0, nt - 1):
#         # Vectorized update for all interior points (no loop needed)
#         u[1:-1, n+1] = u[1:-1, n] + coeff * (u[2:, n] - 2 * u[1:-1, n] + u[:-2, n]) + q(x_values[1:-1], n*dt) * dt

#         # Apply boundary conditions at x = 0 (source term)
#         u[0, n+1] = u[0, n] + coeff * (u[1, n] - 2 * u[0, n] + u[-1, n]) + q(x_values[0], n*dt) * dt

#         # Boundary condition at x = x_max (Dirichlet condition: u -> 0 as x -> infinity)
#         u[-1, n+1] = 0  # Dirichlet condition at the right boundary
    
#     return u, x_values