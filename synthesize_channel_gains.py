import numpy as np
import scipy.io


class RandGenerator:
    """
        The __init__ function initializes a RandGenerator instance. No return.
        num_ues: number of transceiver (one that works full-duplex or half-duplex as transmitter or receiver) pairs
        num_aps: number of access points
        filename: channel gain datasets in .mat file
    """
    def __init__(self, num_ues, num_aps, filename):
        self.num_ues = num_ues
        self.num_aps = num_aps
        self.hrs = range(24 + 1)    # length is 25 since time 0:00 and 24:00 should appear both at head and at tail
        self.rates_per_hr = [0.4, 0.27, 0.13, 0.08, 0.04, 0.04, 0.05, 0.08, 0.21, 0.45, 0.65, 0.83, 0.92,
                             0.94, 0.93, 0.91, 0.90, 0.90, 0.90, 0.98, 0.95, 0.82, 0.70, 0.50, 0.4]
        self.channel_gain = scipy.io.loadmat(filename)

    """
        The generate function generates channel gain datasets under given fading types with path loss. It reshape and 
        uses data from self.channel_gain. Returns channel gain tensor of shape (num_samples, num_aps, num_ues, num_ues)
        and required user rates of shape (num_samples, self.num_ues).
        num_samples: number of complete channel gain instances
        fading_type: fading type to select datasets from self.channel_gain
    """
    def generate(self, num_samples, fading_type):
        # generate locations and compute distances from ues to aps
        ue_locs = np.random.uniform(low=0, high=1, size=(self.num_ues, 2))
        ap_locs = np.random.uniform(low=0, high=1, size=(self.num_aps, 2))
        dist = np.zeros((self.num_ues, self.num_aps))
        for ue in range(self.num_ues):
            for ap in range(self.num_aps):
                dist[ue, ap] = np.maximum(np.linalg.norm(ue_locs[ue] - ap_locs[ap], ord=1), 0.035)

        # compute path loss according to distances. The formula requires d>=0.035 to hold. Omit factor 10 ** (-12.8).
        path_loss = dist ** (-3.76)

        # generate service times and map them to compute required ue rates
        serve_times = np.random.uniform(low=0, high=24, size=(num_samples, self.num_ues))
        ue_rates = np.interp(serve_times, self.hrs, self.rates_per_hr)

        # copy channel gain for given fading type and compute its squared value
        gain = np.abs(self.channel_gain[fading_type]) ** 2
        gain = gain[:num_samples * self.num_aps * self.num_ues ** 2].reshape(
            (num_samples, self.num_aps, self.num_ues, self.num_ues))

        # compute G matrix. Since 20dB - 25dB SINR value is typically recommended for data networks, we could compute
        # the diagonal entries in G must be 10 * num_ues - 18 * num_ues times larger to off-diagonal entries. In fact,
        # this could be achieved by extra techniques like antenna gain and coding gain, etc.
        for i in range(num_samples):
            for j in range(self.num_aps):
                gain[i, j, :, :] = np.matmul(np.diag(path_loss[:, j]) ** 0.5, gain[i, j, :, :])
                gain[i, j, :, :] = np.matmul(gain[i, j, :, :], np.diag(path_loss[:, j]) ** 0.5)
                gain[i, j, :, :] += np.eye(self.num_ues) * gain[i, j, :, :] * 10 ** (22.5 / 20) * self.num_ues
        return [gain, ue_rates]


if __name__ == '__main__':
    rand_generator = RandGenerator(2, 4, 'channel_gain.mat')
    gain, ue_rates = rand_generator.generate(10, 'rayleigh_gain')
    print(gain[0, 0, :])
    print(ue_rates)
