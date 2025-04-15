import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import time


class CrankNicolsonFdm:
    def __init__(self, L, nx, tf, nt, D, q, v0, T):
        """
        Initialize the Crank-Nicolson solver for a general 1D diffusion equation with implicit boundary conditions.

        Parameters:
        L (float): Length of the spatial domain.
        nx (int): Number of spatial grid points.
        tf (float): Final time.
        nt (int): Number of time steps.
        D (callable): Function D(T) returning the diffusion coefficient at temperature T.
        q (callable): Function q(x, t, T) returning the source term at position x and time t.
        v0 (float): Initial concentration coefficient.
        T (callable): Function T(t) returning the temperature at time t.
        """
        # Assign parameters
        self.L = L
        self.nx = nx
        self.tf = tf
        self.nt = nt
        self.D = D
        self.q = q
        self.v0 = v0
        self.T = T

        # Spatial and temporal discretization
        self.dx = L / (nx - 1)
        self.x_grid = np.linspace(0, L, nx)
        self.dt = tf / (nt - 1)
        self.t_grid = np.linspace(0, tf, nt)

        # Initialize concentration using v0 and gaussian_delta
        self.C = np.zeros(self.x_grid.size)
        self.C[0] = self.v0 / self.dx
        self.C_record = np.zeros((self.nt, self.nx))
        self.C_record[0, :] = self.C

    def gaussian_delta(self, x, sigma=0.001, n=6):
        A = 1 / (sigma * np.sqrt(2 * np.pi))  
        return A * np.exp(-((x / sigma) ** (2 * n)))

    def construct_matrices(self, sigma_c):
        nx = self.nx

        main_diag = np.full(nx, 1 + 2 * sigma_c)
        lower_diag = np.full(nx - 1, -sigma_c)
        upper_diag = np.full(nx - 1, -sigma_c)

        # Adjust main diagonal for implicit boundary conditions
        main_diag[0] = 1 + sigma_c
        main_diag[-1] = 1 + sigma_c

        A_c = diags([main_diag, lower_diag, upper_diag], [0, -1, 1], format='csc')

        # Construct B_c
        main_diag_B = np.full(nx, 1 - 2 * sigma_c)
        lower_diag_B = np.full(nx - 1, sigma_c)
        upper_diag_B = np.full(nx - 1, sigma_c)

        B_c = diags([main_diag_B, lower_diag_B, upper_diag_B], [0, -1, 1], format='csc')

        return A_c, B_c

    def crank_nicolson(self):
        start_time = time.time()

        for t in range(1, self.nt):
            # time_in_hours = self.t_grid[t] / 3600.0
            # T_current = self.T(time_in_hours)
            # D_current = self.D(T_current)
            # sigma_c = D_current * self.dt / (2 * self.dx ** 2)

            T_current = 418
            sigma_c = self.D(T_current) * self.dt / (2 * self.dx **2)

            # Reconstruct matrices A_c and B_c with current sigma_c
            A_c, B_c = self.construct_matrices(sigma_c)

            #q = self.q(self.x_grid, self.t_grid[t], T_current) / self.dx
            q = np.zeros(self.x_grid.size)
            q[0] = self.dt * self.q(self.t_grid[t], T_current) / self.dx 

            rhs = B_c.dot(self.C) + q

            # Solve the linear systems
            self.C = spsolve(A_c, rhs)

            self.C_record[t, :] = self.C

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Simulation completed in {execution_time:.2f} seconds.")

    def get_solution(self):
        return self.C_record

    def get_grid(self):
        return self.x_grid, self.t_grid