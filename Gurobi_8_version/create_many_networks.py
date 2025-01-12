# (c) 2017-2019 ETH Zurich, Automatic Control Lab, Joe Warrington, Dominik Ruchti

import numpy as np
import os
import pickle
from drrp.synthetic_system import SyntheticNetwork

def create_network_instances():
    """
    Create a sequence of random network instances and store them in folders in network_data/.
    """
    n_v_array = {
        9: [1], 16: [1], 25: [1], 36: [1],
        64: [1], 100: [1], 225: [1], 400: [1]
    }

    # Ensure the base directory exists
    base_directory = "network_data/"
    os.makedirs(base_directory, exist_ok=True)

    for n in [3]:  # Number of nodes n along the edge of the square grid
        for T in [3]:  # Number of time steps T in planning horizon
            for i in range(1, 3):  # Controls number of instances i and their labels
                # Number of nodes
                N = n * n
                # Total number of bikes in system
                B = N * 5
                # Planning horizon
                W = 3
                # Time Horizon for trips
                K = 2
                # Station capacity
                C_s = np.ceil(1.5 * B / N) * np.ones(N)
                # Vehicle count and capacity
                V_initial = 1
                vehicle_cap = 5
                C_v_initial = vehicle_cap * np.ones(V_initial)
                # Vehicle speed
                vehicle_speed = 100.0 / n * 1.25

                # Name and create target directory for initial V
                network_folder_initial = os.path.join(
                    base_directory, f"N{N:03d}_V{V_initial:02d}_T{T:02d}"
                )
                os.makedirs(network_folder_initial, exist_ok=True)
                filename_initial = os.path.join(network_folder_initial, f"instance_{i:02d}")
                print(f"Creating network: N={N}, V={V_initial}, T={T}, instance {i}")

                # Generate network object
                nw = SyntheticNetwork(N, V_initial, C_s, C_v_initial, B, T, W, K, vehicle_speed)
                with open(f"{filename_initial}.pkl", 'wb') as output:
                    pickle.dump(nw, output, pickle.HIGHEST_PROTOCOL)

                # Iterate over different vehicle counts for the current N
                if N in n_v_array:
                    for V in n_v_array[N]:
                        # Update network object for new V (don't generate new customer demand matrices)
                        C_v = vehicle_cap * np.ones(V)
                        nw.modify_nr_vehicles(V, C_v)

                        # Name and create target directory for updated V
                        network_folder = os.path.join(
                            base_directory, f"N{N:03d}_V{V:02d}_T{T:02d}"
                        )
                        os.makedirs(network_folder, exist_ok=True)
                        filename = os.path.join(network_folder, f"instance_{i:02d}")
                        print(f"Creating network: N={N}, V={V}, T={T}, instance {i}")

                        # Save network object modified for new value of V
                        with open(f"{filename}.pkl", 'wb') as output:
                            pickle.dump(nw, output, pickle.HIGHEST_PROTOCOL)
                else:
                    print(f"Warning: N={N} not found in n_v_array. Skipping additional V configurations.")

if __name__ == "__main__":
    create_network_instances()
