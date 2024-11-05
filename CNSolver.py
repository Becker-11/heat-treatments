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

def D(cur_T):
    nmpercm = 1e7
    return nmpercm **2 * D0 * np.exp(-E_AD / (R * cur_T)) 

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

# Time when temperature changes phases (in seconds)
time_increase_start = 13392  # Time increase starts
time_decrease_start = 30348  # Time decrease starts

# Temperature function T(t)
def get_T(t):
    if t <= time_increase_start:
        return -0.97 * t + 306.92
    elif time_increase_start < t <= time_decrease_start:
        return 62.29 * t + 57.02
    else:
        return 1117.50 * np.exp(-0.17 * t) + 307.66

def construct_matrices(sigma_c, nx):

    # Construct matrices using sparse format
    main_diag = np.full(nx, 1 + 2 * sigma_c)
    lower_diag = np.full(nx - 1, -sigma_c)
    upper_diag = np.full(nx - 1, -sigma_c)

    main_diag[0] = 1 + sigma_c
    main_diag[-1] = 1 + sigma_c

    #Create sparse matrices
    A_c = diags([main_diag, lower_diag, upper_diag], [0, -1, 1], format='csc')

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

    return A_c, B_c 


def crank_nicolson(L, nx, tf, nt):

    start_time = time.time()

    dx = L / (nx - 1)
    x_grid = np.linspace(0, L, nx)

    dt = tf / (nt - 1)
    t_grid = np.linspace(0, tf, nt)

    plt.plot(range(0,45*3600), [get_T(t) for t in range(0,45*3600)])
    plt.show()

    # D_T = D(T) 
    # sigma_c = D_T * dt / (2 * dx**2)

    # Initial condition
    C = v0 * gaussian_delta(x_grid)


    # Preallocate concentration record
    C_record = np.zeros((nt, nx))
    C_record[0, :] = C

    # Time-stepping loop
    for t in range(1, nt):
        cur_T = get_T(t_grid[t])
        sigma_c = D(cur_T) * dt / (2 * dx**2)
        A_c, B_c = construct_matrices(sigma_c, nx)

        rhs = B_c.dot(C) + dt * q(x_grid, t_grid[t])
        #C = np.linalg.solve(A_c, rhs)
        C = spsolve(A_c, rhs)
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

    # # Plot for v(x,t)
    # plt.plot(x_grid, a_v, label=f'analytic v(x,t), t={tf/3600:.1f} h')
    # # Plot for u(x,t)
    # plt.plot(x_grid, a_u, label=f'analytic u(x,t), t={tf/3600:.1f} h')
    # # Combine u(x,t) and v(x,t) for c(x,t)
    # plt.plot(x_grid, a_c, label=f'analytic c(x,t), t={tf/3600:.1f} h')
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


