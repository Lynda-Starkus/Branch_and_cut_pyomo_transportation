# (c) 2017-2019 ETH Zurich, Automatic Control Lab, Joe Warrington, Dominik Ruchti

import numpy as np
from scipy.spatial import distance
from scipy.stats import multivariate_normal
import copy
from matplotlib import pyplot as plt

plt.close('all')


# Generic synthetic demand
class SyntheticDemand:

    def __init__(self, K, O, D, G):

        self.Lag = K
        self.Nr_origins = O
        self.Nr_destinations = D

        self.Grid_dim = G

        g = G / 100.0

        x, y = np.mgrid[0:G:g, 0:G:g]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y

        self.Grid = pos

        self.coordinates = [
            self.Grid[k, r]
            for k in range(len(self.Grid))
            for r in range(self.Grid.shape[0])
        ]

        self.create_origins()
        self.create_destinations()

    def create_origins(self):
        self.Origins = []

        # np.random.seed(self.Nr_origins)
        np.random.seed(None)
        for _ in range(self.Nr_origins):
            x = np.random.randint(0, self.Grid_dim)
            y = np.random.randint(0, self.Grid_dim)
            self.Origins.append(np.array([x, y]))

    def create_destinations(self):
        self.Destinations = []

        # np.random.seed(self.Nr_destinations)
        np.random.seed(None)
        for _ in range(self.Nr_destinations):
            x = np.random.randint(0, self.Grid_dim)
            y = np.random.randint(0, self.Grid_dim)
            self.Destinations.append(np.array([x, y]))

    def create_prob_distribution(self, c, plt_opt=False):
        if c == "O":
            centres = copy.copy(self.Origins)
        elif c == "D":
            centres = copy.copy(self.Destinations)
        else:
            print("Wrong argument!")
            return

        pdf = []
        pdf_tot = np.zeros((self.Grid.shape[0], self.Grid.shape[1]))

        for center in centres:
            mean = center
            v = np.random.randint(1, 4)
            var = v * self.Grid_dim / len(centres)
            pdf.append(multivariate_normal(mean, var))

            pdf_tot += pdf[-1].pdf(self.Grid)

        # Normalize
        pdf_tot += 0.0005
        pdf_cum = pdf_tot / np.sum(pdf_tot)

        if c == "O":
            self.Trip_Origin_PDF = pdf_cum
        elif c == "D":
            self.Trip_Destination_PDF = pdf_cum

        if plt_opt:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            if c == "O":
                ax.plot_surface(
                    self.Grid[:, :, 0],
                    self.Grid[:, :, 1],
                    pdf_cum,
                    cmap='viridis',
                    linewidth=0
                )
                ax.set_xlabel('X axis')
                ax.set_ylabel('Y axis')
                ax.set_zlabel('Z axis')
                plt.show()

                plt.figure()
                plt.contourf(self.Grid[:, :, 0], self.Grid[:, :, 1], pdf_cum)
                plt.title('Trip Origin Distribution')
            elif c == "D":
                ax.plot_surface(
                    self.Grid[:, :, 0],
                    self.Grid[:, :, 1],
                    -1 * pdf_cum,
                    cmap='viridis',
                    linewidth=0
                )
                ax.set_xlabel('X axis')
                ax.set_ylabel('Y axis')
                ax.set_zlabel('Z axis')
                plt.show()

                plt.figure()
                plt.contourf(self.Grid[:, :, 0], self.Grid[:, :, 1], pdf_cum)
                plt.title('Trip Destination Distribution')
                plt.show()

    def draw_samples(self, PDF, nr_samples, plt_opt):
        print(PDF)
        if PDF.size != self.coordinates.__len__():
            # prob = [PDF[x,y] for x in range(PDF.shape[0]) for y in range(PDF.shape[1])]
            prob = np.reshape(PDF, (-1,))  # Flatten the PDF
        else:
            prob = np.reshape(PDF, (-1,))  # Flatten the PDF
            PDF = np.reshape(prob, (int(np.sqrt(len(prob))), int(np.sqrt(len(prob)))))

        print("PROB", prob.shape)

        sample_indices = np.random.choice(np.arange(len(prob)), nr_samples, p=prob)
        samples = [self.coordinates[k] for k in sample_indices]
        samples_x = [sample[0] for sample in samples]
        samples_y = [sample[1] for sample in samples]

        if plt_opt:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(
                self.Grid[:, :, 0],
                self.Grid[:, :, 1],
                PDF,
                cmap='viridis',
                linewidth=0
            )
            ax.scatter(samples_x, samples_y, [0]*len(samples_x))  # Assuming Z=0 for scatter

            plt.figure()
            plt.contourf(self.Grid[:, :, 0], self.Grid[:, :, 1], PDF)
            plt.plot(samples_x, samples_y, 'o')
            plt.show()

        return samples, samples_x, samples_y

    def generate_trips(self, samples, plt_opt=False):

        Dest_PDF = copy.copy(self.Trip_Destination_PDF)
        # prob = [Dest_PDF[x,y] for x in range(Dest_PDF.shape[0]) for y in range(Dest_PDF.shape[1])]
        prob = np.reshape(Dest_PDF, (-1,))  # Flatten the PDF
        # ind = []
        Trips = []

        for sample in samples:
            ind = [distance.euclidean(sample, coord) >= self.Lag * 20 for coord in self.coordinates]
            # ind.append([abs(sample[0]-j[0]+sample[1]-j[1])<=self.Lag*20 for j in self.coordinates])
            prob_temp = copy.copy(prob)
            prob_temp[np.array(ind)] = 0

            pdf_sum = np.sum(prob_temp)
            if pdf_sum == 0:
                # Avoid division by zero
                pdf_cum = prob_temp
            else:
                pdf_cum = prob_temp / pdf_sum

            D_samples, D_samples_x, D_samples_y = self.draw_samples(pdf_cum, 1, False)

            Trips.append((sample, D_samples[-1]))

        if plt_opt:
            plt.figure()
            plt.contourf(self.Grid[:, :, 0], self.Grid[:, :, 1], Dest_PDF)
            for trip in Trips:
                x, y = trip[0]
                dx = trip[1][0] - x
                dy = trip[1][1] - y

                plt.arrow(x, y, dx, dy, width=0.1, head_width=2)
                plt.plot(x, y, 'o', c='black')
                plt.show()
        return Trips


# Generic synthetic grid network
class SyntheticNetwork:
    def __init__(self, N, V, Cs, Cv, B, T, W, K, vehicle_speed):

        self.nr_regions = N
        self.grid_space = int(np.sqrt(N))
        self.nr_vehicles = V

        self.C_s = Cs
        self.C_v = Cv

        self.nr_bikes = B
        self.time_horizon = T
        self.time_window = W
        self.nr_lag_steps = K

        self.centres = None
        self.adjacency_matrix = None
        self.F = None

        self.create_centres()
        self.dist = self.distance_matrix()
        self.create_adjacency_matrix(vehicle_speed)
        self.ds_0 = self.create_initial_bike_distr()
        self.dv_0 = self.create_initial_vehicle_load()
        self.z_0 = self.create_initial_vehicle_distr()

        self.create_demand()

    def __str__(self):
        s = "\n"
        s += f"Network has {self.nr_regions} regions, {self.nr_vehicles} vehicles, and {self.nr_bikes} bikes.\n"
        s += f"Time horizon is {self.time_horizon} steps, and maximum lag is {self.nr_lag_steps} steps.\n"
        s += f"Regions hold {np.mean(self.C_s):.1f} bikes, and vehicles hold {np.mean(self.C_v):.1f} bikes on average."
        return s

    def print_stats(self):
        print("Initial vehicle conditions:")
        N, V = self.nr_regions, self.nr_vehicles
        for v in range(V):
            z_temp = self.z_0[v:(N * V) + v:V, 0].reshape(N)
            start_location = np.argmax(z_temp)
            print(f"  Vehicle #{v:2d} starts in region {start_location:2d} with {int(self.dv_0[v])}/{self.C_v[v]} slots filled.")
        print("Initial bike numbers by region:")
        print(" ", [int(x) for x in self.ds_0.squeeze()])
        if int(np.sum(self.ds_0)) != self.nr_bikes:
            print(f"Warning: {int(np.sum(self.ds_0))} (and not {self.nr_bikes}) bikes in network!")

    def create_centres(self):

        self.centres = []
        step = 100.0 / self.grid_space
        offset = step / 2.0
        for k in range(self.grid_space):
            for j in range(self.grid_space):
                x = j * step + offset
                y = k * step + offset
                self.centres.append(np.array([x, y]))

    def create_adjacency_matrix(self, max_travel_dist):
        N = self.nr_regions
        self.adjacency_matrix = np.zeros((N, N), dtype=int)
        ind = np.where(self.dist <= max_travel_dist)
        self.adjacency_matrix[ind] = 1
        # print("Adjacency matrix:")
        # print(self.adjacency_matrix)

    def create_initial_bike_distr(self):

        # np.random.seed(self.nr_bikes)
        N = self.nr_regions
        B = float(self.nr_bikes)

        r = np.random.randint(0, max(1, int(B / 2)), size=(N))

        x_0 = np.round((r * B) / np.sum(r)) if np.sum(r) > 0 else np.zeros((N))
        x_0 = np.minimum(x_0, self.C_s).astype(int)

        while int(np.sum(x_0)) > self.nr_bikes:
            i = np.random.randint(0, N)
            if x_0[i] > 0:
                x_0[i] -= 1
        while int(np.sum(x_0)) < self.nr_bikes:
            i = np.random.randint(0, N)
            if x_0[i] < self.C_s[i]:
                x_0[i] += 1

        return x_0

    def create_initial_vehicle_distr(self):

        N = self.nr_regions
        V = self.nr_vehicles

        z_0 = np.zeros((N * V, 1))

        available_regions = np.arange(N)

        initial_regions = []
        for v in range(V):
            # select random index out of available regions
            r = np.random.randint(0, len(available_regions))
            initial_region = int(available_regions[r])
            initial_regions.append(initial_region)
            # available_regions = np.delete(available_regions, r)  # Commented out as it may cause issues if regions are reused

            ir = initial_regions[v]
            z_0[ir * V + v, 0] = 1

        return z_0

    def create_initial_vehicle_load(self):

        # np.random.seed(self.nr_vehicles)

        x_0 = np.zeros((self.nr_vehicles, 1))

        return x_0

    def modify_nr_vehicles(self, V, C_v):
        self.nr_vehicles = V
        self.C_v = C_v
        self.z_0 = self.create_initial_vehicle_distr()
        self.dv_0 = self.create_initial_vehicle_load()

    def create_demand(self):

        G = 100.0
        Nr_Origins = 3
        Nr_Destinations = 5

        Demand = SyntheticDemand(self.nr_lag_steps, Nr_Origins, Nr_Destinations, G)

        F = [
            [np.zeros((self.nr_regions, self.nr_regions)) for _ in range(self.nr_lag_steps + 1)]
            for _ in range(self.time_horizon + 1)
        ]

        total_demand = 0

        for k in range(self.time_window):

            Demand.create_origins()
            Demand.create_destinations()

            for t in range(self.time_horizon // self.time_window):
                Demand.create_prob_distribution("O", False)
                Demand.create_prob_distribution("D", False)

                n = max(int(np.random.normal(0.15 * self.nr_bikes, 0.075 * self.nr_bikes)), 0)
                print("NR_bikes", Demand.Trip_Origin_PDF)
                origin_samples, ox, oy = Demand.draw_samples(Demand.Trip_Origin_PDF, n, False)

                Trips = Demand.generate_trips(origin_samples, False)

                for trip in Trips:
                    Lag = int(distance.euclidean(trip[0], trip[1]) / 20)
                    o = int(trip[0][0] // (G / np.sqrt(self.nr_regions))) + \
                        int(np.sqrt(self.nr_regions) * (int(trip[0][1] // (G / np.sqrt(self.nr_regions)))))
                    d = int(trip[1][0] // (G / np.sqrt(self.nr_regions))) + \
                        int(np.sqrt(self.nr_regions) * (int(trip[1][1] // (G / np.sqrt(self.nr_regions)))))

                    if 0 <= o < self.nr_regions and 0 <= d < self.nr_regions and 0 <= Lag <= self.nr_lag_steps:
                        F[k * (self.time_horizon // self.time_window) + t + 1][Lag][o, d] += 1
                    else:
                        print("Error: Invalid trip indices or lag.")

                total_demand += len(Trips)
                print(f"  {len(Trips)} trips created at time {k * (self.time_horizon // self.time_window) + t + 1}.")

        print(f"  Total demand is {total_demand} trips; {total_demand / self.nr_bikes:.2f} trips/bike.")

        self.F = F

    def distance_matrix(self):

        dist_matrix = np.zeros((self.nr_regions, self.nr_regions))

        for k in range(self.nr_regions):
            for l in range(self.nr_regions):
                dist_matrix[k, l] = distance.euclidean(self.centres[k], self.centres[l])

        return dist_matrix

    def repos_penalty(self, t, P):
        dist = self.dist.copy()
        np.fill_diagonal(dist, 0)

        return P * dist

    def trip_reward(self, t, k, R1):

        dist = self.dist.copy()
        np.fill_diagonal(dist, R1)

        return dist
