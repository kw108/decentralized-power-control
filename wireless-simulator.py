import numpy as np
import itertools
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from synthesize_channel_gains import RandGenerator

warnings.filterwarnings('error')


class AccessPoint:
    """
        The __init__ function initializes an AccessPoint instance. No return.
        num_ues: number of transceiver (one that works full-duplex or half-duplex as transmitter or receiver) pairs
        gain: channel gain of shape (num_ues, num_ues)
        noise_dbm: noise level at receivers measure in dBm of shape (num_ues,)
        eps: epsilon to decide when to stop iteration
    """
    def __init__(self, num_ues, gain, noise_dbm, eps):
        self.num_ues = num_ues
        self.noise = np.ones(shape=(num_ues,)) * (10 ** (noise_dbm / 10))
        self.eps = eps
        self.G = gain
        self.F = gain - np.eye(num_ues) * gain

    """
        The compute_capacity function computes channel capacity given power vector and channel gain. Returns channel 
        capacity vector of shape (num_ues,).
        p: power allocation vector at transmitters with current access point of shape (num_ues,)
    """
    def compute_capacity(self, p):
        sinr = np.divide(np.matmul(self.G - self.F, p), np.matmul(self.F, p) + self.noise)
        return np.log(sinr + 1)

    """
        The check_feasibility function checks feasibility for current access point with given bit rate and channel 
        gain. Returns True or False.
        rate: bit rate assignment vector at receivers with current access point of shape (num_ues,)
    """
    def check_feasibility(self, rate):
        d = np.diag(np.divide(np.e ** rate - 1, np.diag(self.G)))    # bit rate using log_e
        try:
            np.linalg.eigvals(np.matmul(d, self.F))
        except np.linalg.LinAlgError:
            return False
        if not (np.less(np.linalg.eigvals(np.matmul(d, self.F)), np.ones_like(rate))).all():
            return False
        return True

    """
        The fp_iterate function uses a fixed-point iteration to update power vector allocation p. Returns converged 
        power allocation vector of shape (num_ues,)
        rate: bit rate assignment vector at receivers with current access point of shape (num_ues,)
    """
    def fp_iterate(self, rate):
        if not self.check_feasibility(rate):
            return np.ones_like(self.noise) * np.inf
        d = np.diag(np.divide(np.e ** rate - 1, np.diag(self.G)))  # bit rate using log_e
        p = np.ones_like(self.noise)
        p_last = np.zeros_like(self.noise)
        while np.dot(p - p_last, p - p_last) > self.eps:
            p_last = p
            p = np.matmul(self.F, p) + self.noise
            p = np.matmul(d, p)
        return p


class Simulator:
    """
        The __init__ function initializes a Simulator instance. No return.
        num_ues: number of transceiver (one that works full-duplex or half-duplex as transmitter or receiver) pairs
        num_aps: number of (cooperative) access points
        gain: channel gain of shape (num_ues, num_ues)
        rate: total required bit rate vector at receivers of shape (num_ues,)
        noise_dbm: noise level at receivers measure in dBm
        eps: epsilon to decide when to stop iteration
    """
    def __init__(self, num_ues, num_aps, gain, rate, noise_dbm, eps):
        self.num_ues = num_ues
        self.num_aps = num_aps
        self.gain = gain
        self.rate = rate
        self.eps = eps
        self.APs = [AccessPoint(num_ues, gain[i], noise_dbm, eps) for i in range(num_aps)]

    """
        The solve_opt function solves for optimal total power required to satisfy (individual) bit rate requirements. 
        It should always invoke check_feasibility before computing allocated power vector. Returns optimal power 
        allocation matrix of shape (num_ues, num_aps).
        rate: bit rate allocation matrix of shape (num_ues, num_aps)
    """
    def solve_opt(self, rate):
        opt_p = np.zeros_like(rate)
        for i in range(num_aps):
            opt_p[:, i] = self.APs[i].fp_iterate(rate[:, i])
        return opt_p

    """
        The grid_search function solves for optimal total power required to satisfy (sum) bit rate requirements. This 
        could be done with, e.g., a naive complete grid search. Returns optimal power allocation matrix of shape
        (num_ues, num_aps).
        num_folds: increment in amounts of 1 / num_folds of each hyperparameter to tune
        verbose_mode: show progress if grid search takes very long time
    """
    def grid_search(self, num_folds, verbose_mode=False):
        samples = itertools.combinations_with_replacement(range(num_folds + 1), self.num_aps - 1)
        samples = [sorted(s) for s in samples]
        samples = [[sj - si for si, sj in zip([0] + s, s + [num_folds])] for s in samples]
        cartesian = [samples] * self.num_ues
        opt_p = np.ones(shape=(self.num_ues, self.num_aps)) * np.Inf
        cnt = 0
        s = len(list(itertools.product(*cartesian)))
        for t in itertools.product(*cartesian):
            if verbose_mode:
                if cnt == int(s * 0.8):
                    print('80 percent finished')
                elif cnt == int(s * 0.6):
                    print('60 percent finished')
                elif cnt == int(s * 0.4):
                    print('40 percent finished')
                elif cnt == int(s * 0.2):
                    print('20 percent finished')
            cnt += 1
            m = np.array(t) / num_folds
            """
            if (m < np.ones_like(m) * 1 / num_folds).any():     # skip current combination if any bit rate is 0
                continue
            """
            p = self.solve_opt(m)
            if p.sum() < opt_p.sum():
                opt_p = p
        return opt_p

    """
        The heuristic_search function optimizes power in the proposed heuristic approach from the paper. Returns 
        optimal power allocation matrix of shape (num_ues, num_aps).
        num_iterations: number of outer iterations in heuristic search (specified for plot)
    """
    def heuristic_search(self, num_iterations=20):
        # initialization
        p = np.ones(shape=(self.num_ues, self.num_aps))
        x = np.ones_like(p)
        v = np.ones_like(p)
        # w = np.random.uniform(low=0, high=1, size=(self.num_ues, self.num_aps))
        w = np.ones_like(p)
        w = w / w.sum(axis=1)[..., np.newaxis]
        r = np.matmul(np.diag(self.rate), w)
        r_last = np.zeros_like(r)
        ps = []
        # outer loop for r
        for _ in range(num_iterations):     # while ((r - r_last) ** 2).sum() > eps:
            r_last = r
            # inner loop for p
            p_last = np.zeros_like(p)
            while ((p - p_last) ** 2).sum() > self.eps:
                p_last = p
                p = self.solve_opt(r)
                if np.isinf(p.sum()):
                    print('rate not feasible')
            # inner loop for x -- could be incorporated into inner loop for p
            x_last = np.zeros_like(x)
            for i in range(num_aps):
                while ((x[:, i] - x_last[:, i]) ** 2).sum() > self.eps:
                    x_last[:, i] = x[:, i]
                    A = np.linalg.inv(np.diag(np.diag(self.APs[i].G)))
                    A = np.matmul(A, self.APs[i].F)
                    A = np.matmul(A, np.diag(np.e ** r[:, i] - 1))
                    x[:, i] = np.matmul(A, x[:, i]) + 1
            # update v
            for k1 in range(len(r)):
                for k2 in range(len(r[0])):
                    try:
                        v[k1, k2] = r[k1, k2] * np.e ** r[k1, k2] / (np.e ** r[k1, k2] - 1) * p[k1, k2] * x[k1, k2]
                    except RuntimeWarning:
                        v[k1, k2] = np.e ** r[k1, k2] * p[k1, k2] * x[k1, k2]
                        continue
                    if np.isnan(r[k1, k2] * np.e ** r[k1, k2] / (np.e ** r[k1, k2] - 1)) * p[k1, k2] * x[k1, k2]:
                        v[k1, k2] = np.e ** r[k1, k2] * p[k1, k2] * x[k1, k2]
                    else:
                        v[k1, k2] = r[k1, k2] * np.e ** r[k1, k2] / (np.e ** r[k1, k2] - 1) * p[k1, k2] * x[k1, k2]
            # update w
            w = r / v
            w = w / w.sum(axis=1)[..., np.newaxis]
            r = np.matmul(np.diag(r.sum(axis=1)), w)
            ps.append(self.solve_opt(r))
        return [ps, r]


if __name__ == '__main__':
    # set global parameters
    num_ues = 8
    num_aps = 4
    noise_dbm = -60
    eps = 0.001
    filename = 'channel_gain.mat'
    fading_type = 'rayleigh_gain'
    num_samples = 50    # number of channel gain and user rate samples
    num_iterations = 100     # number of outer iterations in heuristic search (specified for plot)
    num_folds = 5  # number of folds used in grid search for global optimal solution
    verbose_mode = False    # show progress if grid search takes very long time

    # generate channel gain data according to global parameters
    rand_generator = RandGenerator(num_ues, num_aps, filename)
    gains, ue_rates = rand_generator.generate(num_samples, fading_type, offset=0)

    # run wireless simulations with the 0-th channel gain and user rate sample
    simulator = Simulator(num_ues, num_aps, gains[0], ue_rates[0], noise_dbm, eps)

    # obtain power allocation under equal rate assignment
    rate_equ = (ue_rates[0] / num_aps)[..., np.newaxis] * np.ones(shape=(1, num_aps))
    equ_p = simulator.solve_opt(rate_equ)
    print('Power under equal bit rate alloc: %.10f' % equ_p.sum(), end=', ')

    # obtain power allocation under grid search by testing all possible rate assignments
    # opt_p = simulator.grid_search(num_folds, verbose_mode=False)
    # print('Power under grid search in bit rate: %.10f' % opt_p.sum(), end=', ')

    # obtain power allocation under randomly initialized heuristic search
    heu_ps, heu_r = simulator.heuristic_search(num_iterations=20)
    print('Power under randomly initialized heuristic search: %.10f' % heu_ps[-1].sum(), end='.\n')

    # repeat for heuristic search due to random initialization
    # pm of shape (num_aps, num_samples, num_iterations)
    pm = np.zeros(shape=(num_aps, num_samples, num_iterations))
    for i in range(num_samples):
        simulator = Simulator(num_ues, num_aps, gains[i], ue_rates[i], noise_dbm, eps)
        ps, r = simulator.heuristic_search(num_iterations=num_iterations)
        for j in range(num_aps):
            for k in range(num_iterations):
                pm[j, i, k] = ps[k].sum(axis=0)[j]
    np.save('pm', pm)
