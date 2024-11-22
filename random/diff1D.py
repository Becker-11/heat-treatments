import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
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


def gaussian_delta(x, sigma=0.00689, n=6):
    A = 1 / (sigma * np.sqrt(2 * np.pi))  
    return A * np.exp(-((x / sigma) ** (2 * n)))

def D(T):
    nmpercm = 1e7
    return nmpercm **2 * D0 * np.exp(-E_AD / (R * T)) 

def k(T):
    return A_D * np.exp(-E_a / (R * T))

def q(x, t):
    return u0 * k(T) * np.exp(-k(T)*t) * gaussian_delta(x)

def analytic_u(x,t):
    int_val, err = quad(u_integrand, 0, t, args=(t, x), epsabs=1e-10, epsrel=1e-10)
    return int_val

def u_integrand(s, t, x):
    return (u0 * k(T) * np.exp(-k(T) * t) / np.sqrt(np.pi * D(T) * (t - s))) * np.exp(-x**2 / (4 * D(T) * (t - s)))

def solver_FE_simple(a, f, L, dt, F, T):
    """
    Simplest expression of the computational algorithm
    using the Forward Euler method and explicit Python loops.
    For this method F <= 0.5 for stability.
    """
    start_time = time.time()

    Nt = int(round(T/float(dt)))
    t = np.linspace(0, Nt*dt, Nt+1)   # Mesh points in time
    dx = np.sqrt(a*dt/F)
    Nx = int(round(L/dx))
    x = np.linspace(0, L, Nx+1)       # Mesh points in space
    # Make sure dx and dt are compatible with x and t
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    u   = np.zeros(Nx+1)
    u_n = np.zeros(Nx+1)

    # Set initial condition u(x,0) = I(x)
    for i in range(0, Nx+1):
        u_n[i] = q(x[i], 0)

    for n in range(0, Nt):
        # Compute u at inner mesh points
        u[1:-1] = u_n[1:-1] + F*(u_n[0:-2] - 2*u_n[1:-1] + u_n[2:]) + dt*f(x[1:-1], t[n])

        # Insert boundary conditions
        #u[0] = 0;  u[Nx] = 0 # Dirichlet
        u[0] = u[1]; u[Nx] = 0 # Nuemann

        # Switch variables before next step
        #u_n[:] = u  # safe, but slow
        u_n, u = u, u_n

    end_time = time.time()
    run_time = start_time - end_time

    return u_n, x, t, start_time  # u_n holds latest u

def plot_u(u, x, Nt):
    # Plot both functions for comparison
    plt.figure(figsize=(10, 6))
    plt.plot(x, u, label='u(x,t)', color='blue')
    #plt.plot(x, [analytic_u(i,Nt) for i in x])
    plt.plot(x, q(x,0))
    

    plt.title('u(x,t)')
    plt.xlabel('x (nm)')
    plt.ylabel('Concentration % nm')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():

    L = 300
    time = 45 * 3600
    F = D(T)
    Nx = 300
    Nt = 150000
    x_values = np.linspace(0, L, Nx+1)    # mesh points in space
    dx = x_values[1] - x_values[0]
    t_values = np.linspace(0, time, Nt+1)    # mesh points in time
    dt = t_values[1] - t_values[0]
    F = D(T)*dt/dx**2

    u, x_values, t, cpu_time = solver_FE_simple(D(T), q, L, dt, F, T)

    plot_u(u, x_values, Nt)



if __name__=="__main__":
    main()