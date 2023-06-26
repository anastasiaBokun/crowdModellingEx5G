import numpy as np

def read_data(verbose: bool = False):
    # Source: stackoverflow
    filename = 'data_task5/MI_timesteps.txt'
    array = []

    try:
        with open(filename, 'r') as current_file:
            # Reads each line of the file, and creates a 1d list of each point
            # Skip the first line, because it contains the names of the columns
            skip = True
            for line in current_file.readlines():
                if skip:
                    skip = False
                    continue

                # Get the first line
                values = line.split(' ')
                # Create a vector where we store the 10 values of a line
                new_data = np.zeros(10)

                # For each of the 10 values
                for i in range(10):
                    # Store the value in the new vector
                    x = int(values[i])
                    new_data[i] = x

                # Add the new vector to the whole collection of data
                array.append(new_data)
    except IOError:
        print("Something went wrong when attempting to read file.")

    array = np.array(array)
    if verbose:
        print("The old shape is {shape}.".format(shape=array.shape))

    # As per the exercise sheet we remove the first 1000 timesteps
    array = array[1000:, 1:]
    if verbose:
        print("The new shape is {shape}.".format(shape=array.shape))

    return array

# Calculating phi of x for a single column
def phi_x(x, x_l, epsilon):
    result = np.exp(-np.square(x_l - x) / epsilon)
    return result

# Calculating phi of x for a whole matrix
def phi_X(X, epsilon, L):
    N = X.shape[0]
    result = np.zeros((N, L))
    for i in range(L):
        phi_i = phi_x(X, X[i], epsilon)
        result[:, i] = phi_i
    return result

# Finding the coefficient matrix, as in the case of the linear variant
def find_C(X, epsilon, L, F):
    N = X.shape[0]
    Phi = phi_X(X, epsilon, L)
    C = np.linalg.lstsq(Phi, F[:N], rcond = 1)[0]
    return C

# This function plots the 2 red helper points before and after the maximum of teh investigated concave region.
def period_helper(i_left, i_right, i_start, ax1, speeds):
    ax1.plot(i_left, speeds[i_left], marker="o", markersize=5, color="red")
    ax1.plot(i_right, speeds[i_right], marker="o", markersize=5, color="red")
    i_end = i_left + np.argmax(speeds[i_left:i_right])
    ax1.plot(i_end, speeds[i_end], marker="o", markersize=5, color="orange")
    period = i_end-i_start

    return i_start, period, i_end

# After observing the graph, we have manually chosen the red points in order to obtain the maximum of the concave regions.
# Once the index of the maximum is known we can find the intervals for each of the periods by substracting consecutive indices for the maximum values.
def check_periods(ax1, speeds):
    periods=[]
    i_start_s = []
    ax1.plot(0, speeds[0], marker="o", markersize=5, color="orange")
    i_start, period,i_end = period_helper(1900,2000,0,ax1, speeds)
    # print(period)
    periods.append(period)
    i_start_s.append(i_start)
    i_start, period,i_end = period_helper(3900,4000,i_end,ax1, speeds)
    # print(period)
    periods.append(period)
    i_start_s.append(i_start)
    i_start, period,i_end = period_helper(5900,6000,i_end,ax1, speeds)
    # print(period)
    periods.append(period)
    i_start_s.append(i_start)
    i_start, period,i_end = period_helper(7900,8000,i_end,ax1, speeds)
    # print(period)
    periods.append(period)
    i_start_s.append(i_start)
    i_start, period,i_end = period_helper(9900,10000,i_end,ax1, speeds)
    # print(period)
    periods.append(period)
    i_start_s.append(i_start)
    i_start, period,i_end = period_helper(11900,12050,i_end,ax1, speeds)
    #print(i_start)
    periods.append(period)
    i_start_s.append(i_start)
    i_start_s.append(i_end)

    period = np.average(periods[3])

    return int(period), periods, i_start_s