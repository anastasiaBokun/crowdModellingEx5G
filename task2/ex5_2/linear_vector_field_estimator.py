import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class LinearVectorFieldEstimator:
    def __init__(self, x0_data_file, x1_data_file, dt=0.1):
        """
        Initializes the LinearVectorFieldEstimator.

        Args:
            x0_data_file (str): File path of the x0 data.
            x1_data_file (str): File path of the x1 data.
            dt (float): Time step for finite difference estimation.
        """
        self.x0_data = np.loadtxt(x0_data_file)
        self.x1_data = np.loadtxt(x1_data_file)
        self.dt = dt
        self.v_data = (self.x1_data - self.x0_data) / self.dt
        self.A_hat = None

    def estimate_matrix_A(self):
        """
        Estimates the matrix A using the x0 and v_data.

        Returns:
            None
        """
        self.A_hat = np.linalg.lstsq(self.x0_data, self.v_data, rcond=None)[0]

    def solve_linear_system(self, T, x0):
        """
        Solves the linear system x_dot = A_hat * x with the given initial conditions.

        Args:
            T (float): End time for simulation.
            x0 (ndarray): Initial point.

        Returns:
            solution (scipy.integrate.OdeSolution): Solution of the linear system.
        """
        def linear_system(t, x):
            return self.A_hat.dot(x)

        solution = solve_ivp(linear_system, [0, T], x0, method='RK45')
        return solution

    def compute_mean_squared_error(self, solution):
        """
        Computes the mean squared error between the estimated points and x1_data.

        Args:
            solution (scipy.integrate.OdeSolution): Solution of the linear system.

        Returns:
            mse (float): Mean squared error.
        """
        x1_estimated = solution.y[:, -1]
        mse = np.mean(np.square(x1_estimated - self.x1_data))
        return mse

    def solve_extended_system(self, Tend, x0_extended):
        """
        Solves the extended linear system x_dot = A_hat * x for an extended period of time.

        Args:
            Tend (float): End time for extended simulation.
            x0_extended (ndarray): Initial point for extended simulation.

        Returns:
            solution_extended (scipy.integrate.OdeSolution): Solution of the extended linear system.
        """
        def linear_system_extended(t, x):
            return self.A_hat.dot(x)

        solution_extended = solve_ivp(linear_system_extended, [0, Tend], x0_extended, method='RK45')
        return solution_extended

    def plot_trajectory(self, solution):
        """
        Plots the trajectory of the linear system.

        Args:
            solution (scipy.integrate.OdeSolution): Solution of the linear system.

        Returns:
            None
        """
        plt.figure(figsize=(8, 6))
        plt.plot(solution.y[0], solution.y[1], 'b-', label='Trajectory')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Trajectory')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_phase_portrait(self, solution_extended):
        """
        Plots the phase portrait of the extended linear system.

        Args:
            solution_extended (scipy.integrate.OdeSolution): Solution of the extended linear system.

        Returns:
            None
        """
        plt.figure(figsize=(8, 6))
        plt.quiver(solution_extended.y[0][:-1], solution_extended.y[1][:-1],
                   np.diff(solution_extended.y[0]), np.diff(solution_extended.y[1]),
                   scale_units='xy', angles='xy', scale=1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Phase Portrait')
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.grid(True)
        plt.show()
