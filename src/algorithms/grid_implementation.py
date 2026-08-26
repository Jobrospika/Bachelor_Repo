''' Module for the implementation of the grid-based safe optimization algorithm. '''
import math
from abc import ABC
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import torch

#################################################################################################
# SafeBo Class
#################################################################################################


class SafeBO(ABC):
    '''
    Base class for Safe Optimization algorithms.
    '''
    def __init__(self, config, gp):
        self.gp_model = gp  # GP model
        # bounds of the input space D for example:  [(0, 1), (0, 1)]
        self.bounds = config["bounds"]
        self.seed_set = config["seed_set"]   # S_0
        self.safe_set = self.seed_set   # S_t
        self.safety_threshold = config["safety_threshold"]   # h
        self.lipschitz_constant = config["lipschitz_constant"]  # L
        self.E = config["E"]  # noise bound
        self.beta = config["beta"]  # beta
        self.X = self.gp_model.train_inputs[0]
        self.Y = self.gp_model.train_targets
        if len(self.Y.shape) == 0:
            self.Y = self.Y.unsqueeze(0)

##################################################################################################
# SafeOptGrid Class
##################################################################################################


class SafeOptGrid(SafeBO):

    def __init__(self, config, gp):
        super().__init__(config, gp)

        self.points_per_axis = config["points_per_axis"]
        self.grid = linearly_spaced_combinations_torch(
            self.bounds, self.points_per_axis)

        # determine closest point in grid to safe set
        self.safe_index = find_index(self.grid, self.safe_set)
        self.expander_index = None
        self.maximizer_index = None
        self.G = None
        self.M = None

        # get index of each linearly spaced point in the grid
        self.grid_index = torch.arange(self.points_per_axis**len(self.bounds))
        self.initialize_C()
        self.initialize_Q()
        self.l_t = self.C[:, 0]
        self.u_t = self.C[:, 1]

    def initialize_C(self):
        """
        initialize C as vector of zeros with the same size as the grid 
        """
        self.C = torch.ones(self.points_per_axis**len(self.bounds), 2)
        self.C[:, 0] = -np.inf
        self.C[:, 1] = np.inf
        self.C[self.safe_index, 0] = self.safety_threshold
        # how to handle the seed set?

    def initialize_Q(self):
        """
        initialize confidence interval Q
        """
        self.Q = torch.zeros(self.points_per_axis**len(self.bounds), 2)
        self.Q[:, 0] = -np.inf
        self.Q[:, 1] = np.inf

    def optimize(self):
        """
        one iteration of the safe optimization algorithm
        """
        self.update_C()
        self.calculate_S()
        self.calculate_G()
        self.calculate_M()
        print(f"  [n_obs={self.X.shape[0]}] safe_index={self.safe_index.shape[0]}  "
              f"G={0 if self.G is None else self.G.shape[0]}  "
              f"M={0 if self.M is None else self.M.shape[0]}")
        x_next = self.optimize_acquisition_function()
        return x_next

    def update_gp(self):
        """
        predict new Q
        """
        self.gp_model.eval()
        pred = self.gp_model(self.grid)
        self.Q[:, 0] = pred.mean - self.beta*pred.variance.sqrt()
        self.Q[:, 1] = pred.mean + self.beta*pred.variance.sqrt()

    def update_C(self):
        """
        update confidence interval C
        """
        # this would be the correct way to update C, but it breaks if the uncertainty bounds are wrong
        # self.C[:,0] = torch.maximum(self.C[:,0], self.Q[:,0])
        # self.C[:,1] = torch.minimum(self.C[:,1], self.Q[:,1])
        self.C[:, 0] = self.Q[:, 0]
        self.C[:, 1] = self.Q[:, 1]
        self.l_t = self.C[:, 0]
        self.u_t = self.C[:, 1]

    def calculate_S(self):
        """
        determine safe set S
        """

        d = torch.cdist(self.grid, self.safe_set, p=2)
        self.l_t = self.C[:, 0]
        self.u_t = self.C[:, 1]

        # safety_condition = self.l_t[:, None] - self.lipschitz_constant * d >= self.safety_threshold
        safety_condition = self.l_t[:, None] >= self.safety_threshold
        safe_indices = torch.nonzero(safety_condition)

        self.safe_index = torch.unique(safe_indices[:, 0])

        self.safe_set = self.grid[self.safe_index]

    def calculate_G(self):
        """
        determine expander set G
        """

        self.G = None
        if self.safe_index.shape[0] < self.grid_index.shape[0]:
            mask = torch.ones(self.grid_index.shape[0], dtype=torch.bool)
            mask[self.safe_index] = False
            grid_without_safe_set = self.grid[mask]

            # Calculate distances
            d = torch.cdist(self.safe_set, grid_without_safe_set, p=2)
            # Select the corresponding u_t values
            u_t = self.C[self.safe_index, 1]

            # Reshape u_t for broadcasting
            u_t = u_t.unsqueeze(1)  # Shape becomes [N, 1]

            # Vectorized condition checking
            expander_condition = (
                u_t - self.lipschitz_constant * d) >= self.safety_threshold

            # Find the indices where the condition is true and get unique indices
            expander_indices = torch.nonzero(expander_condition)
            unique_expander_indices = torch.unique(expander_indices[:, 0])

            # If there are unique indices, select corresponding grid points
            if unique_expander_indices.numel() > 0:
                self.G = self.grid[self.safe_index[unique_expander_indices]]
            else:
                self.G = torch.empty(0, len(self.bounds))
        else:
            print("No new points to add to G")
            self.G = torch.empty(0, len(self.bounds))

    def calculate_M(self):
        """
        determine maximizer set M
        """

        l_t = self.C[:, 0]
        u_t = self.C[:, 1]

        # Initialize self.M to None or an empty tensor as per your requirements
        self.M = None

        # Check if safe_index is not empty
        if self.safe_index.numel() > 0:
            # Find max of l_t in safe set
            l_max = torch.max(l_t[self.safe_index])

            # Vectorized comparison
            condition = u_t[self.safe_index] - l_max > 0

            # Use boolean indexing to select the relevant rows from the grid
            self.M = self.grid[self.safe_index][condition]

            # Add a dimension if self.M is not None and has no second dimension
            if self.M is not None and self.M.ndim == 1:
                self.M = self.M.unsqueeze(0)
        else:
            print("safe_index is empty, no elements to process.")

    def optimize_acquisition_function(self):
        """
        maximize acquisition function on G and M
        """
        if self.G is None:
            G_and_M = self.M
        elif self.M is None:
            G_and_M = self.G
        else:
            G_and_M = torch.cat((self.G, self.M), dim=0)
        if G_and_M.shape[0] == 0:
            raise Exception("No new points to add to GP")
        variance = 2*self.beta*self.gp_model(G_and_M).variance.sqrt()
        max_ind = torch.argmax(variance)
        return G_and_M[max_ind].unsqueeze(0)

    def add_data_to_gp(self, X, Y):
        """
        Add data to the GP.
        """
        current_X = self.X
        current_Y = self.Y
        self.X = torch.cat((current_X, X), dim=0)
        self.Y = torch.cat((current_Y, Y), dim=0)
        self.gp_model.set_train_data(self.X, self.Y, strict=False)
        return self.gp_model

    def generate_plot(self, observe_data, ax=None):
        """
        Generate plot for 1D function
        
        Parameters: 
            observe_data: function to observe the data
            ax: axis to plot the data
        
        Returns:
            fig: figure
            ax: axis
        
        """

        test = torch.linspace(0, 1, 100).unsqueeze(1)
        pred = self.gp_model(test)
        best_mean = self.get_current_best_mean_x().unsqueeze(0)

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(test, observe_data(test))
        ax.plot(test, pred.mean.detach())
        ax.fill_between(
            self.grid.detach().flatten(),
            self.l_t.detach(),
            self.u_t.detach(),
            alpha=0.2,
        )

        ax.scatter(self.gp_model.train_inputs[0], self.gp_model.train_targets)
        ax.scatter(self.safe_set, torch.ones(len(self.safe_set)) *
                   self.safety_threshold-1, color="green", marker="s")
        ax.scatter(best_mean, observe_data(
            best_mean), color="yellow", marker="o")
        try:
            ax.scatter(self.G, torch.ones(len(self.G)) *
                       self.safety_threshold-1.4, color="pink", marker="s")
        except:
            pass
        try:
            ax.scatter(self.M, torch.ones(len(self.M)) *
                       self.safety_threshold-1.2, color="blue", marker="s")
        except:
            pass
        ax.scatter(self.seed_set, observe_data(self.seed_set), color="blue")
        ax.plot(test, torch.ones(len(test)) *
                self.safety_threshold, color="red")
        return fig, ax

    def get_current_best_mean_x(self):
        self.gp_model.eval()
        pred = self.gp_model(self.safe_set)
        return self.safe_set[torch.argmax(pred.mean)]

################################################################################################
# RealBetaSafeOpt Class
################################################################################################


class RealBetaSafeOpt(SafeOptGrid):
    ''' 
    Real Beta Safe Optimization Algorithm 
    '''
    def __init__(self, config, gp, beta_config):
        super().__init__(config, gp)
        self.beta_config = beta_config
        self.beta = self.calculate_beta()

    def calculate_beta(self):
        """
        determine beta
        """
        beta_config = self.beta_config
        K = self.gp_model.covar_module(self.X, self.X).to_dense()
        # self.beta = beta_config["B"] + (beta_config["R"]  / beta_config["lamb"]  ** 0.5) * math.sqrt(math.log(torch.linalg.det(K*lamb_hat/beta_config["lamb"] + lamb_hat * torch.eye(len(self.X)))) - 2 * math.log(beta_config["delta"]))
        self.beta = beta_config["B"]+beta_config["R"]/math.sqrt(beta_config["lamb"]) * math.sqrt(
            2*math.log(1/beta_config["delta"]*torch.linalg.det(K/beta_config["lamb"] + torch.eye(len(self.X)))))

    def update_gp(self):
        """
        predict new Q
        """
        self.calculate_beta()
        self.gp_model.eval()
        pred = self.gp_model(self.grid)
        self.Q[:, 0] = pred.mean - self.beta*pred.variance.sqrt()
        self.Q[:, 1] = pred.mean + self.beta*pred.variance.sqrt()

#####################################################################################################
# Losbo Class
#####################################################################################################

class Losbo(SafeOptGrid):
    """
    LOSBO algorithm
    """
    def __init__(self, config, gp):
        super().__init__(config, gp)

    def calculate_S(self):
        """
        determine safe set S
        """

        d = torch.cdist(self.X, self.grid, p=2)
        self.l_t = self.C[:, 0]
        self.u_t = self.C[:, 1]

        # Calculate the condition for all elements
        # Reshape self.Y and l_t for broadcasting if necessary
        condition = (self.Y.unsqueeze(1) - self.E -
                     self.lipschitz_constant * d) >= self.safety_threshold

        # Use boolean indexing to find the indices where the condition is true
        safe_indices = torch.nonzero(condition)

        self.safe_index = torch.unique(safe_indices[:, 1])

        self.safe_set = self.grid[self.safe_index]


def linearly_spaced_combinations_torch(bounds, num_samples):
    """
    Return 2-D tensor with all linearly spaced combinations within the bounds.

    Parameters
    ----------
    bounds: sequence of tuples
        The bounds for the variables, [(x1_min, x1_max), (x2_min, x2_max), ...]
    num_samples: integer or array_like
        Number of samples to use for every dimension. Can be a constant if
        the same number should be used for all, or an array to fine-tune
        precision. Total number of data points is num_samples ** len(bounds).

    Returns
    -------
    combinations: 2-d tensor
        A 2-d tensor. If d = len(bounds) and l = prod(num_samples) then it
        is of size l x d, that is, every row contains one combination of
        inputs.
    """
    num_vars = len(bounds)

    if not isinstance(num_samples, (list, tuple)):
        num_samples = [num_samples] * num_vars

    if len(bounds) == 1:
        return torch.linspace(bounds[0][0], bounds[0][1], num_samples[0]).unsqueeze(1)

    # Create linearly spaced test inputs
    inputs = [torch.linspace(b[0], b[1], n)
              for b, n in zip(bounds, num_samples)]

    # Compute all combinations using itertools.product
    combinations = list(product(*inputs))

    # Convert combinations to a 2-D tensor
    return torch.tensor(combinations, dtype=torch.float32)


def find_index(grid, safe_set):
    """
    find index of the closest point in grid to safe set
    """
    dist = torch.cdist(grid, safe_set)
    min_idx = torch.argmin(dist, dim=0)
    return min_idx
