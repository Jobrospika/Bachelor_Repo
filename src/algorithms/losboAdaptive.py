import numpy as np
import torch

from src.algorithms.grid_implementation import SafeBO, Losbo


class LosboAdaptive(Losbo):

    def __init__(self, config, gp):
        SafeBO.__init__(self, config, gp)

        self.dim = len(self.bounds)
        self.n_safe_samples = config.get("n_safe_samples", 2000)
        self.n_frontier_samples = config.get("n_frontier_samples", 2000)

        self.lows = torch.tensor([b[0] for b in self.bounds], dtype=torch.float32)
        self.highs = torch.tensor([b[1] for b in self.bounds], dtype=torch.float32)
        ranges = self.highs - self.lows

        base_local_std = config.get("local_std", 0.05)
        if isinstance(base_local_std, (int, float)):
            base_local_std = ranges * base_local_std
        else:
            base_local_std = torch.as_tensor(base_local_std, dtype=torch.float32)

        lipschitz_scaling = config.get("lipschitz_scaling", True)
        if lipschitz_scaling:
            lipschitz_reference = config.get("lipschitz_reference", 1.0)
            scale_factor = lipschitz_reference / self.lipschitz_constant
            self.local_std = base_local_std * scale_factor
        else:
            self.local_std = base_local_std

        self._candidate_seed = config.get("candidate_seed", 0)

        self.expander_index = None
        self.maximizer_index = None
        self.G = None
        self.M = None

        self._refresh_candidate_pool()
        self.initialize_C()
        self.initialize_Q()
        self.l_t = self.C[:, 0]
        self.u_t = self.C[:, 1]

    def _refresh_candidate_pool(self):
        torch.manual_seed(self._candidate_seed)
        self._candidate_seed += 1

        obs_safe_mask = self.Y >= self.safety_threshold
        anchor = self.X[obs_safe_mask] if obs_safe_mask.any() else self.seed_set

        idx_safe = torch.randint(0, anchor.shape[0], (self.n_safe_samples,))
        interior = anchor[idx_safe] + torch.randn(self.n_safe_samples, self.dim) * (self.local_std * 0.3)
        interior = torch.clamp(interior, self.lows, self.highs)

        frontier_base = self.safe_set if self.safe_set.shape[0] > 0 else anchor
        idx_frontier = torch.randint(0, frontier_base.shape[0], (self.n_frontier_samples,))
        frontier = frontier_base[idx_frontier] + torch.randn(self.n_frontier_samples, self.dim) * self.local_std
        frontier = torch.clamp(frontier, self.lows, self.highs)

        self.grid = torch.cat([anchor, interior, frontier], dim=0)
        self.grid_index = torch.arange(self.grid.shape[0])

        self._n_provisional_safe = anchor.shape[0] + interior.shape[0]

    def initialize_C(self):
        n = self.grid.shape[0]
        self.C = torch.ones(n, 2)
        self.C[:, 0] = -np.inf
        self.C[:, 1] = np.inf

        self.C[:self._n_provisional_safe, 0] = self.safety_threshold

    def initialize_Q(self):
        n = self.grid.shape[0]
        self.Q = torch.zeros(n, 2)
        self.Q[:, 0] = -np.inf
        self.Q[:, 1] = np.inf

    def update_gp(self):
        self._refresh_candidate_pool()
        self.initialize_C()
        self.initialize_Q()
        self.gp_model.eval()
        pred = self.gp_model(self.grid)
        self.Q[:, 0] = pred.mean - self.beta * pred.variance.sqrt()
        self.Q[:, 1] = pred.mean + self.beta * pred.variance.sqrt()


def set_up_algorithm_adaptive(algorithm, config, model):
    '''
    Extended version of set_up_algorithm that also recognizes "losbo_adaptive".
    Merge this branch into your existing set_up_algorithm rather than keeping
    two separate factory functions.
    '''
    if algorithm == "losbo_adaptive":
        opt = LosboAdaptive(config=config, gp=model)
        #opt.update_gp()
        return opt
    raise NotImplementedError(f"Unknown algorithm: {algorithm}")