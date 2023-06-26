import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def load_data(str_file):
    """
    Load data from txt file into a numpy array

    args:
        str_file: if 0 or 1, load known file for the exercise
                  if str, load specific file

    returns:
        d (np.ndarray): of shape (n_data, space_dim) - loaded data
    """
    if str_file == 0:
        d = np.loadtxt("nonlinear_vectorfield_data_x0.txt")
    elif str_file == 1:
        d = np.loadtxt("nonlinear_vectorfield_data_x1.txt")
    else:
        d = np.loadtxt(str_file)
    return d

def apply_estimator(df, X, Y, n_dt, t_max=1):
    """
    Predicts X(t) with df = V the vector field, and compares the comparaison to a given position in the futur

    args
        df (lambda t, x):
            vector field - each points' derivative in time
        X (np.ndarray):
            data points at t=0
        Y (np.ndarray):
            ground truth at t=dt (unknown dt)
        n_dt (int):
            number of samples in time (from t=0 to t=t_max)
        t_max (opt. scalar):
            prediction's duration

    returns:
        sols (np.ndarray):
            Trajectories of each data points
        mses (np.ndarray):
            mean square error for each timestamp
    """

    # predict trajectories for each data points
    sols = []
    for x in X:
        sol = solve_ivp(df, [0, t_max], x, t_eval=np.linspace(0, t_max, n_dt))
        sols.append(sol.y)

    sols = np.asarray(sols)

    # compute mean square error for each timestamp
    mses = []
    for i in range(n_dt):
        mses.append(mean_squared_error(Y, sols[:,:,i]))
    mses = np.asarray(mses)

    return sols, mses


def plot_vector_field(dX, min_bound, max_bound, n_1d_samples):
    """Plots the vector field induced by dX (2d)

    Args:
        dX (function): 
            Takes sample points in 2d space, returns its tangeant vector
        min_bound (list of 2 scalars):
            min bounds in 2d space of the plot
        max_bound (list of 2 scalars): 
            max bounds in 2d space of the plot
        n_1d_samples (int): 
            number of samples to compute the vector field (in 1d)
            total number of samples is therefore squared
            
    Returns:
        U1, U2 (np.ndarray):
            vectorfield as a 2d meshgrid
        X1, X2 (np.ndarray):
            sample points as a 2d meshgrid
    """
    X1, X2 = np.meshgrid(np.linspace(min_bound[0], max_bound[0], n_1d_samples), np.linspace(min_bound[1], max_bound[1], n_1d_samples))
    X = np.stack([X1, X2]).reshape((2, n_1d_samples**2)).T
    
    V = dX(0, X)
    U1, U2 = V.T.reshape(2, n_1d_samples, n_1d_samples)
    
    _, axs = plt.subplots()
    vfn = np.linalg.norm(np.array([U1, U2]), axis=0)
    axs.streamplot(X1, X2, U1, U2, density=1, linewidth = 5*vfn / vfn.max(), color=vfn)
    axs.set_title("Predicted Vector Field")
    
    return U1, U2, X1, X2
