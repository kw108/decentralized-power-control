import numpy as np
import itertools
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings('error')


class AccessPoint:
    """
    The __init__ function initializes an AccessPoint instance. No return.
    num_tx_pairs: number of transceiver (one that works full-duplex or half-duplex as transmitter or receiver) pairs
    noise_dbm: noise level at receivers measure in dBm
    channel_type: channel fading type
    eps: epsilon to decide when to stop iteration
    """
    def __init__(self, num_tx_pairs, noise_dbm, channel_type, eps):
        self.num_tx_pairs = num_tx_pairs
        self.noise = np.ones(shape=(num_tx_pairs,)) * (10 ** (noise_dbm / 10))  # measured in mWatt
        self.eps = eps
        self.G = np.zeros(shape=(num_tx_pairs, num_tx_pairs))   # initialized as a placeholder
        if channel_type == 'Gaussian':
            self.G = np.random.standard_normal(size=[num_tx_pairs, num_tx_pairs])
            self.G += np.diag(np.diag(self.G)) * 10
            self.G = np.abs(self.G)
        self.F = self.G - np.diag(np.diag(self.G))

    """
    The compute_capacity function computes channel capacity given power vector and channel gain. Returns channel 
    capacity vector of shape (num_tx_pairs,).
    p: power allocation vector at transmitters with current access point of shape (num_tx_pairs,)
    """
    def compute_capacity(self, p):
        sinr = np.divide(np.matmul(self.G - self.F, p), np.matmul(self.F, p) + self.noise)
        return np.log(sinr + np.ones_like(sinr))

    """
    The check_feasibility function checks feasibility for current access point with given bit rate and channel gain. 
    Returns True or False.
    bit_rate: bit rate assignment vector at receivers with current access point of shape (num_tx_pairs,)
    """
    def check_feasibility(self, bit_rate):
        d = np.diag(np.divide(np.e ** bit_rate - np.ones_like(bit_rate), np.diag(self.G)))    # bit rate using log_e
        try:
            np.linalg.eigvals(np.matmul(d, self.F))
        except np.linalg.LinAlgError:
            print('bit_rate is')
            print(bit_rate)
            print('np.e ** bit_rate')
            print(np.e ** bit_rate)
            print('np.diag(self.G)')
            print(np.diag(self.G))
            print('d is')
            print(d)
            exit(1)
        if (np.less(np.linalg.eigvals(np.matmul(d, self.F)), np.ones_like(bit_rate))).all():
            return True
        else:
            # print('The largest eigenvalue not smaller than 1.0 and so system not feasible.')
            return False

    """
    The fp_iterate function uses a fixed-point iteration to update power vector allocation p. Returns converged power 
    allocation vector of shape (num_tx_pairs,)
    """
    def fp_iterate(self, bit_rate):
        if not self.check_feasibility(bit_rate):
            return np.ones_like(self.noise) * np.inf
        d = np.diag(np.divide(np.e ** bit_rate - np.ones_like(bit_rate), np.diag(self.G)))  # bit rate using log_e
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
    num_tx_paris: number of transceiver (one that works full-duplex or half-duplex as transmitter or receiver) pairs
    num_access_points: number of (cooperative) access points
    bit_rate: total required bit rate vector at receivers of shape (num_tx_pairs,)
    noise_dbm: noise level at receivers measure in dBm
    channel_type: channel fading type
    eps: epsilon to decide when to stop iteration
    """
    def __init__(self, num_tx_pairs, num_access_points, bit_rate, noise_dbm, channel_type, eps):
        self.num_tx_pairs = num_tx_pairs
        self.num_access_points = num_access_points
        self.bit_rate = bit_rate
        self.APs = []
        for _ in range(num_access_points):
            self.APs.append(AccessPoint(num_tx_pairs, noise_dbm, channel_type, eps))

    """
    The solve_opt function solves for optimal total power required to satisfy (individual) bit rate requirements. It 
    should always invoke check_feasibility before computing allocated power vector. Returns optimal power allocation 
    matrix of shape (num_tx_pairs, num_access_points).
    bit_rate: bit rate allocation matrix of shape (num_tx_pairs, num_access_points)
    """
    def solve_opt(self, bit_rate):
        opt_p = np.zeros_like(bit_rate)
        for i in range(num_access_points):
            opt_p[:, i] = self.APs[i].fp_iterate(bit_rate[:, i])
        return opt_p

    """
    The grid_search function solves for optimal total power required to satisfy (sum) bit rate requirements. This could 
    be done with, e.g., a naive complete grid search. Returns optimal power allocation matrix of shape
    (num_tx_pairs, num_access_points).
    num_folds: increment in amounts of 1/num_folds of each hyperparameter to tune (i.e., (sum) bit rate per transceiver)
    """
    def grid_search(self, num_folds):
        samples = itertools.combinations_with_replacement(range(num_folds + 1), self.num_access_points - 1)
        samples = [sorted(s) for s in samples]
        samples = [[sj - si for si, sj in zip([0] + s, s + [num_folds])] for s in samples]
        cartesian = []
        for _ in range(self.num_tx_pairs):
            cartesian.append(samples)
        opt_p = np.ones(shape=(self.num_tx_pairs, self.num_access_points)) * np.Inf
        s = len(list(itertools.product(*cartesian)))
        cnt = 0
        for t in itertools.product(*cartesian):
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
            # if (m < np.ones_like(m) * 1 / num_folds).any():     # skip current combination if any bit rate is 0
                # continue
            p = self.solve_opt(m)
            if p.sum() < opt_p.sum():
                opt_p = p
        return opt_p

    """
    The heuristic_search function optimizes power in the proposed heuristic approach from the paper. Returns optimal 
    power matrix of shape (num_tx_pairs, num_access_points).
    eps: epsilon to decide when to stop iteration
    """
    def heuristic_search(self, eps):
        # 1)
        p = np.ones(shape=(self.num_tx_pairs, self.num_access_points))
        x = np.zeros_like(p)
        v = np.zeros_like(p)
        # 2)
        r = np.ones_like(p)
        r_last = np.zeros_like(p)
        while np.multiply(r - r_last, r - r_last).sum() > eps:
            r_last = r
            for i in range(num_access_points):
                r[:, i] = self.APs[i].compute_capacity(p[:, i])
            w = r / r.sum(axis=1)[..., np.newaxis]
            # 3)
            p_last = np.zeros_like(p)
            while np.multiply(p - p_last, p - p_last).sum() > eps:
                p_last = p
                for i in range(num_access_points):
                    p[:, i] = np.divide(np.matmul(np.diag(self.bit_rate), p[:, i]),
                                        np.matmul(np.diag(self.bit_rate), np.ones_like(w[:, i]) - w[:, i]) + r[:, i])
                # 4)
                for i in range(num_access_points):
                    x[:, i] = np.matmul(np.linalg.inv(self.APs[i].G - self.APs[i].F),
                                        np.matmul(self.APs[i].F, (np.e ** r[:, i] - 1) * x[:, i]))\
                              + np.ones_like(x[:, i])
                v = r * np.e ** r / (np.e ** r - 1) * p * x
            # 5) go to while clause in 3)
            # 6)
            w = r / v
            w = w / w.sum(axis=1)[..., np.newaxis]
            for i in range(num_access_points):
                r[:, i] = self.APs[i].compute_capacity(p[:, i])
        # 7) go to while clause in 2)
        # print(r.sum(axis=1))
        # print(self.bit_rate)
        return p

    """
    The heuristic_search_2 function optimizes power in the proposed heuristic approach from the paper. Returns optimal 
    power allocation matrix of shape (num_tx_pairs, num_access_points).
    eps: epsilon to decide when to stop iteration
    """
    def heuristic_search_2(self, eps):
        # 1)
        p = np.ones(shape=(self.num_tx_pairs, self.num_access_points))
        x = np.zeros_like(p)
        v = np.zeros_like(p)
        # 2)
        w = np.random.uniform(low=0, high=1, size=(self.num_tx_pairs, self.num_access_points))
        w = w / w.sum(axis=1)[..., np.newaxis]
        r = np.matmul(np.diag(self.bit_rate), w)
        r_last = np.zeros_like(p)
        ps = []
        for _ in range(10):   # while np.multiply(r - r_last, r - r_last).sum() > eps:
            r_last = r
            # 3)
            p_last = np.zeros_like(p)
            while np.multiply(p - p_last, p - p_last).sum() > eps:
                p_last = p
                p = self.solve_opt(r)
                if np.isinf(p.sum()):
                    print('rate not feasible')
            # 4)
            x_last = np.ones_like(x)
            for i in range(num_access_points):
                while np.multiply(x[:, i] - x_last[:, i], x[:, i] - x_last[:, i]).sum() > eps:
                    x_last[:, i] = x[:, i]
                    try:
                        np.linalg.inv(self.APs[i].G - self.APs[i].F)
                    except np.linalg.LinAlgError:
                        print(self.APs[i].G - self.APs[i].F)
                        exit(1)
                    x[:, i] = np.matmul(np.linalg.inv(self.APs[i].G - self.APs[i].F),
                                        np.matmul(self.APs[i].F, (np.e ** r[:, i] - 1) * x[:, i]))\
                              + np.ones_like(x[:, i])
            try:
                v = r * np.e ** r / (np.e ** r - 1) * p * x
            except RuntimeWarning:
                return np.ones_like(p) * np.NaN
            # 5) go to while clause in 3)
            # 6)
            w = r / v
            w = w / w.sum(axis=1)[..., np.newaxis]
            r = np.matmul(np.diag(r.sum(axis=1)), w)
            ps.append(self.solve_opt(r))
            # print(self.solve_opt(r).sum(), end=', ')
        # 7) go to while clause in 2)
        print()
        return ps


if __name__ == '__main__':
    # setting global parameters
    num_tx_pairs = 2
    num_access_points = 6
    bit_rate = np.random.uniform(low=0, high=1, size=(num_tx_pairs,))
    print('bit rate per transceiver pair: ', end='')
    print(bit_rate, end='\n\n')
    noise_dbm = -60
    channel_type = 'Gaussian'
    eps = 0.001
    num_folds = 5

    # running wireless simulations
    simulator = Simulator(num_tx_pairs, num_access_points, bit_rate, noise_dbm, channel_type, eps)
    bit_rate_equ = (bit_rate / num_access_points)[..., np.newaxis] * np.ones(shape=(1, num_access_points))
    equ_p = simulator.solve_opt(bit_rate_equ)
    opt_p = simulator.grid_search(num_folds)
    print('Power under equal bit rate alloc: %.10f, optimal power: %.10f (in mWatt)'
          % (equ_p.sum(), opt_p.sum()), end='\n\n')

    # repeating for heuristic search
    cnt = 0
    pm = np.zeros(shape=(simulator.num_access_points, 10, 10))
    for i in range(100):
        if cnt >= 10:
            break
        ps = simulator.heuristic_search_2(eps)
        if np.isnan(np.array(ps).sum()):
            continue
        for j in range(simulator.num_access_points):
            for k in range(10):
                pm[j, cnt, k] = ps[k].sum(axis=0)[j]
        cnt += 1
    print(pm[0, :, :].sum(axis=0))
    np.save('pm', pm)
