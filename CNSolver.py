import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.integrate import quad
from scipy.sparse.linalg import spsolve

import time


np.set_printoptions(precision=3)

# Model Parameters
v0 = 10          # Initial condition coefficient
u0 = 1000              # Source term coefficient
D0 = 0.0138
E_a = 131e3
A_D = 0.9e9   
E_AD = 111.53e3
R = 8.314
T = 418.15    # 280C = 553.15K

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


def crank_nicolson(L, nx, tf, nt):

    start_time = time.time()

    dx = L / (nx - 1)
    x_grid = np.linspace(0, L, nx)

    dt = tf / (nt - 1)
    t_grid = np.linspace(0, tf, nt)

    D_T = D(T) 
    sigma_c = D_T * dt / (2 * dx**2)

    # Initial condition
    C = v0 * gaussian_delta(x_grid)

    # Construct matrices using sparse format
    main_diag = np.full(nx, 1 + 2 * sigma_c)
    lower_diag = np.full(nx - 1, -sigma_c)
    upper_diag = np.full(nx - 1, -sigma_c)

    main_diag[0] = 1 + sigma_c
    main_diag[-1] = 1 + sigma_c

    #Create sparse matrices
    A_c_sparse = diags([main_diag, lower_diag, upper_diag], [0, -1, 1], format='csc')

    # A_c = np.diagflat([-sigma_c for i in range(nx-1)], -1) +\
    # np.diagflat([1.+sigma_c]+[1.+2.*sigma_c for i in range(nx-2)]+[1.+sigma_c]) +\
    # np.diagflat([-sigma_c for i in range(nx-1)], 1)

    # print("Dense A_c:")
    # print(A_c)

    # print("Sparse A_c (converted to dense):")
    # print(A_c_sparse.toarray())

    # print(np.allclose(A_c, A_c_sparse.toarray()))


    # Construct B_c 
    main_diag_B = np.full(nx, 1 - 2 * sigma_c)
    lower_diag_B = np.full(nx - 1, sigma_c)
    upper_diag_B = np.full(nx - 1, sigma_c)

    B_c = diags([main_diag_B, lower_diag_B, upper_diag_B], [0, -1, 1], format='csc')

    # Preallocate concentration record
    C_record = np.zeros((nt, nx))
    C_record[0, :] = C

    # Time-stepping loop
    for t in range(1, nt):
        rhs = B_c.dot(C) + dt * q(x_grid, t_grid[t])
        #C = np.linalg.solve(A_c, rhs)
        C = spsolve(A_c_sparse, rhs)
        C_record[t, :] = C

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Simulation completed in {execution_time:.2f} seconds.")


    return C, x_grid



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
    L = 300. # Spatial domain in nm
    nx = 830 # Number of spatial points

    tf = 45*3600  # Maximum time in seconds (point in time we integrate to numerically)
    nt = 6000 # Number of time steps

    C, x_grid = crank_nicolson(L, nx, tf, nt)
    plot_concentrations(C, x_grid, tf)

    
    

if __name__ == "__main__":
    main()




# def crank_nicolson(L, nx, tf, nt):

#     start_time = time.time()

#     dx = L / (nx - 1)
#     x_grid = np.linspace(0, L, nx)

#     dt = tf / (nt - 1)
#     t_grid = np.linspace(0, tf, nt)

#     D_T = D(T)  # Ensure D(T) is defined and returns a scalar
#     sigma_c = D_T * dt / (2 * dx**2)

#     # Initial condition
#     C = v0 * gaussian_delta(x_grid)  # Ensure gaussian_delta is vectorized

#     # Preallocate concentration record
#     C_record = np.zeros((nt, nx))
#     C_record[0, :] = C

#     # Construct tridiagonal coefficients for A_c
#     a = -sigma_c * np.ones(nx - 1)  # Sub-diagonal (a_1 to a_{n-1})
#     b = (1 + 2 * sigma_c) * np.ones(nx)  # Main diagonal (b_1 to b_n)
#     c = -sigma_c * np.ones(nx - 1)  # Super-diagonal (c_1 to c_{n-1})

#     # Construct B_c diagonals
#     d_lower = sigma_c * np.ones(nx - 1)
#     d_main = (1 - 2 * sigma_c) * np.ones(nx)
#     d_upper = sigma_c * np.ones(nx - 1)

#     # Time-stepping loop
#     for t in range(1, nt):
#         # Compute RHS: B_c.dot(C) + dt * q(x_grid, t_grid[t])
#         # Since B_c is tridiagonal, we can compute B_c.dot(C) efficiently
#         rhs = np.zeros(nx)
#         # Compute B_c.dot(C)
#         rhs[1:-1] = d_lower[1:] * C[:-2] + d_main[1:-1] * C[1:-1] + d_upper[:-1] * C[2:]
#         # Handle boundaries implicitly (assuming zero-flux or natural boundaries)
#         rhs[0] = d_main[0] * C[0] + d_upper[0] * C[1]
#         rhs[-1] = d_lower[-1] * C[-2] + d_main[-1] * C[-1]
#         # Add source term
#         rhs += dt * q(x_grid, t_grid[t])  # Ensure q is vectorized

#         # Solve tridiagonal system using Thomas algorithm
#         C = thomas_algorithm(a, b, c, rhs)
#         C_record[t, :] = C

#     end_time = time.time()
#     execution_time = end_time - start_time
#     print(f"Simulation completed in {execution_time:.2f} seconds.")

#     return C, x_grid


# def thomas_algorithm(a, b, c, d):
#     """
#     Solves a tridiagonal system Ax = d where:
#     - a: sub-diagonal (length n-1)
#     - b: main diagonal (length n)
#     - c: super-diagonal (length n-1)
#     - d: right-hand side (length n)
#     """
#     n = len(b)
#     c_prime = np.zeros(n - 1)
#     d_prime = np.zeros(n)
    
#     # Modify the first coefficient
#     c_prime[0] = c[0] / b[0]
#     d_prime[0] = d[0] / b[0]
    
#     # Forward sweep
#     for i in range(1, n - 1):
#         denom = b[i] - a[i - 1] * c_prime[i - 1]
#         c_prime[i] = c[i] / denom
#         d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denom
        
#     # Last element of forward sweep
#     denom = b[-1] - a[-1] * c_prime[-1]
#     d_prime[-1] = (d[-1] - a[-1] * d_prime[-2]) / denom
    
#     # Backward substitution
#     x = np.zeros(n)
#     x[-1] = d_prime[-1]
#     for i in range(n - 2, -1, -1):
#         x[i] = d_prime[i] - c_prime[i] * x[i + 1]
        
#     return x
