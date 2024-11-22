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
        D (float): Diffusion coefficient (assumed constant).
        q (callable): Function q(x, t) returning the source term at position x and time t.
        v0 (float): Initial concentration coefficient.
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

        # Compute sigma_c
        self.sigma_c = D(T) * self.dt / (2 * self.dx ** 2)

        # Initialize concentration using v0 and gaussian_delta
        self.C = self.v0 * self.gaussian_delta(self.x_grid)
        self.C_record = np.zeros((self.nt, self.nx))
        self.C_record[0, :] = self.C

        # Construct matrices
        self.A_c, self.B_c = self.construct_matrices()

    def gaussian_delta(self, x, sigma=0.001, n=6):
        A = 1 / (sigma * np.sqrt(2 * np.pi))  
        return A * np.exp(-((x / sigma) ** (2 * n)))

    def construct_matrices(self):
        nx = self.nx
        sigma_c = self.sigma_c

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
            q = self.q(self.x_grid, self.t_grid[t])

            rhs = self.B_c.dot(self.C) + self.dt * q

            # Solve the linear systems
            self.C = spsolve(self.A_c, rhs)

            self.C_record[t, :] = self.C

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Simulation completed in {execution_time:.2f} seconds.")

    def get_solution(self):
        return self.C_record

    def get_grid(self):
        return self.x_grid, self.t_grid
