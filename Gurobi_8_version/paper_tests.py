# (c) 2017-2019 ETH Zurich, Automatic Control Lab, Joe Warrington, Dominik Ruchti

"""
Stochastic approximation approach for value function approximation in Dynamic Repositioning and
Rerouting Problem (DRRP). Based on Powell, Ruszczynksi, and Topaloglu, "Learning Algorithms for
Separable Approximations of Discrete Stochastic Optimization Problems", Math of OR, 2004.
"""

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import pickle
import numpy as np
import os
from itertools import product
from drrp.stoch_approx import SAModel

def ensure_directories(directories):
    """
    Ensure that the specified directories exist. Create them if they do not.
    
    Parameters:
    - directories (list of str): List of directory paths to ensure existence.
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_network_instance(filename):
    """
    Load a network instance from a pickle file.
    
    Parameters:
    - filename (str): Path to the pickle file.
    
    Returns:
    - nw: Loaded network instance.
    """
    with open(filename, 'rb') as input_file:
        nw = pickle.load(input_file)
    return nw

def main():
    # Ensure required directories exist
    ensure_directories(['output', 'vfs'])

    # Load system data
    cost_params = {
        'lost demand cost': 1.0,
        'vehicle movt cost': 1e-3,
        'load cost': 1e-3,
        'unload cost': 1e-3,
        'lost bike cost': 20,
        'created bike cost': 20,
        'lost demand cost spread low': 0.5,
        'lost demand cost spread high': 1.5
    }

    alg_params = {
        'n_iter': 50,
        'ss_rule': 'PRT',
        'max_dev': 10,
        '1k_const': 40,
        'ss_const': 0.2,
        'relax_s1': 'All z',
        'plot_every': 10,
        'cost_eval_samples': 100,
        'eval_cost_every': 5,
        'final_sol': True,
        'final_sol_method': 'exact',
        'save_iter_models': False,
        'eval_cost_k': [10, 20, 50],
        'random_s1': False,
        'nominal_s2': False
    }

    T = 3  # Number of time steps

    # Define the mapping of number of regions to vehicle counts
    n_v_array = {
        9: [1],
        16: [1],
        25: [1],
        36: [1],
        64: [1],
        100: [1],
        225: [1],
        400: [1]
    }

    # Define the number of regions to process
    regions_to_process = [9]  # Currently only processing N=9; adjust as needed

    # Define the range of instances
    instance_range = range(1, 3)  # Instances 1 through 10

    # Iterate through all combinations of N, instance, and V
    for N, i in product(regions_to_process, instance_range):
        # Iterate through all vehicle counts for the current N
        for V in n_v_array.get(N, []):
            print(f"Testing N={N}, V={V}, T={T}, instance {i}")

            # Construct the filename for the current instance
            filename = f"network_data/N{N:03d}_V{V:02d}_T{T:02d}/instance_{i:02d}.pkl"

            if os.path.exists(filename):
                try:
                    # Load the network instance
                    nw = load_network_instance(filename)

                    # Initialize vehicle distribution
                    nw.dv_0 = np.zeros((V, 1))

                    # Set up the SAModel
                    s = SAModel(nw, cost_params, alg_params, T, i, label='_regular')

                    # Run approximation algorithm and output results
                    s.eval_no_action_cost()
                    s.approx()
                    # Uncomment the following line if you wish to run the integer-only method
                    # s.integer_only()

                except Exception as e:
                    print(f"An error occurred while processing {filename}: {e}")
            else:
                print(f"File {filename} does not exist! Skipping.")

if __name__ == "__main__":
    main()
