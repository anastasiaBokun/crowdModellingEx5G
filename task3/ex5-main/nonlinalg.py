import numpy as np
from numpy.linalg import lstsq, norm
from scipy.integrate import solve_ivp


def nonlinear_approximation(X : np.ndarray, Y : np.ndarray, Dt, L, e=None, use_e_squared = True):
    """
    Approximate the vector field V such that Y = X + dt*V using radial basis functions

    args:
        X (np.ndarray):
            of shape (n_data, n_components) - data points at t=0
        Y (np.ndarray):
            of shape (n_data, n_components) - data points at t=dt
        Dt (scalar):
            supposed delta time between Y and X
        L (int):
            Number of radial functions used (exact iff L is a perfect square number)
        e (opt. scalar):
            bandwidth
            if None : e = .05 * dataset's diameter
        use_e_quared (bool):
            if True, e**2 will be used as the denomitaor in the radial functions
        
    returns:
        C (np.ndarray):
            matrix of shape (L, n_components) such that v = phi(X) @ C.T
        samples_points (list of n_components dim points):
            all x_l points used in radial functions
        e (scalar):
            given e, or computed if None was given
    """
    n_data, n_comp = X.shape
    if X.shape != Y.shape:
        raise ValueError(f"{X.shape} different of {Y.shape}")
    if n_comp != 2:
        raise NotImplementedError()

    ### sample L uniform points in the 2d space
    sample_points = []
    sqrt_L = np.sqrt(L) # number of points in 1 dim

    min_bound = np.array([X.min(axis=0), Y.min(axis=0)]).min(axis=0) 
    max_bound = np.array([X.max(axis=0), Y.max(axis=0)]).max(axis=0)

    xs = np.linspace(min_bound[0], max_bound[0], int(sqrt_L))
    ys = np.linspace(min_bound[1], max_bound[1], int(sqrt_L))

    # sample points in a 2d meshgrid
    # As L is typically <1000, simple for loops are ok
    for x in xs:
        for y in ys:
            x_l = np.array([x, y])
            sample_points.append(x_l)

    if e is None:
        e = .05 * norm(max_bound - min_bound) # e = .05 * dataset's diameter

    ### Minimize least_square error
    applied_phi = apply_phi(sample_points, X, e, use_e_squared) # embedding of the points
    V = estimate_vector_field(X, Y, Dt) # A rough estimate of the vector field

    C, res, r, s = lstsq(applied_phi, V, rcond=None)

    return C, sample_points, e


def radial_basis_fun(x_l, x, e, use_e_squared = True):
    """
    Apply a radial function on x

    args:
        x_l (np.ndarray):
            sample point of the radial function
        x (np.ndarray):
            of shape (n_data, n_comp) or (n_comp) - data point(s)
        e (scalar):
            bandwidth
        use_e_quared (bool):
            if True, e**2 will be used as the denomitaor in the radial functions

    returns:
        (np.ndarray): of shape (n_data, n_comp) or (n_comp) - applied radial function
    
    """
    e = e**2 if use_e_squared else e
    if x.ndim > 1: # ndim == 2
        return np.exp(-norm(x_l - x, axis=1) / e)
    return np.exp(-norm(x_l - x) / e)
    
def estimate_vector_field(X, Y, Dt):
    """
    vector field estimation where Y = X + dt*V
    TODO: use better estimator

    args
        X - data points at t=0
        Y - data points at t=dt
        Dt - estimation of dt
    returns
        V - vector field
    """
    return (Y - X) / Dt

def apply_phi(sample_points, X, e, use_e_squared):
    """
    Apply a nonlinear transformation of X using radial functions with same bandwidth

    args
        sample_points (list):
            list of sample points defining each radial function
        X (np.ndarray):
            data points
        e:
            bandwidth
        use_e_quared (bool):
            if True, e**2 will be used as the denomitaor in the radial functions
    
    returns:
        (np.ndarray): of shape (n_data, L) - embedding of X using radial functions
    
    """
    applied_phi = [radial_basis_fun(x_l, X, e, use_e_squared) for x_l in sample_points]
    return np.asarray(applied_phi).T
