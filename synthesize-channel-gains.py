import numpy as np
import scipy.io
import pickle


class RandGenerator:
    hrs = range(24 + 1)  # length is 25 since time 0:00 and 24:00 should appear both at head and at tail
    rates_per_hr = [0.400, 0.265, 0.127, 0.084, 0.032, 0.024, 0.046, 0.110, 0.257, 0.450, 0.715, 0.845, 0.928,
                    0.942, 0.920, 0.908, 0.892, 0.882, 0.888, 0.934, 0.926, 0.825, 0.733, 0.598, 0.400]

    """
        The __init__ function initializes a RandGenerator instance. No return.
        num_ues: number of transceiver (one that works full-duplex or half-duplex as transmitter or receiver) pairs
        num_aps: number of access points
        filename: channel gain datasets in .mat file
    """
    def __init__(self, num_ues, num_aps, filename_gain, filename_rate, filename_locs):
        self.num_ues = num_ues
        self.num_aps = num_aps
        self.channel_gain = scipy.io.loadmat(filename_gain)
        self.ue_rates = scipy.io.loadmat(filename_rate)
        self.locs = scipy.io.loadmat(filename_locs)

    """
        The generate function generates channel gain datasets under given fading types with path loss. It reshape and 
        uses data from self.channel_gain. Returns channel gain tensor of shape (num_samples, num_aps, num_ues, num_ues) 
        and required user rates of shape (num_samples, self.num_ues).
        num_samples: number of channel gain and user rate samples
        fading_type: fading type to select datasets from self.channel_gain
        offset: starting point of the generated channel gain data to take
    """
    def generate(self, num_samples, fading_type, offset=0):
        # generate locations and compute distances from ues to aps
        ap_locs = np.abs(self.locs['ap_locs'])
        ue_locs = np.abs(self.locs['ue_locs'])
        try:
            assert num_samples * self.num_aps * 2 + offset < len(ap_locs)
            assert num_samples * self.num_ues * 2 + offset < len(ue_locs)
            ap_locs = ap_locs[offset: num_samples * self.num_aps * 2 + offset].reshape((num_samples, self.num_aps, 2))
            ue_locs = ue_locs[offset: num_samples * self.num_ues * 2 + offset].reshape((num_samples, self.num_ues, 2))
        except AssertionError:
            print('Trying to generate more samples than allowed.')

        # generate service times and map them to compute required ue rates
        serve_times = np.random.uniform(low=0, high=24, size=(num_samples, self.num_ues))
        ue_rates = np.interp(serve_times, self.hrs, self.rates_per_hr)

        # copy channel gain for given fading type and compute its squared value
        gain = np.abs(self.channel_gain[fading_type]) ** 2
        try:
            assert num_samples * self.num_aps * self.num_ues ** 2 + offset < len(gain)
            gain = gain[offset: num_samples * self.num_aps * self.num_ues ** 2 + offset].reshape(
                (num_samples, self.num_aps, self.num_ues, self.num_ues))
        except AssertionError:
            print(np.shape(gain))
            gain = np.random.choice(np.ravel(gain), size=(num_samples, self.num_aps, self.num_ues, self.num_ues),
                                    replace=True)

        rate = np.abs(self.ue_rates['ue_rates'])
        try:
            assert num_samples * self.num_ues + offset < len(rate)
            rate = rate[offset: num_samples * self.num_ues + offset].reshape((num_samples, self.num_ues))
        except AssertionError:
            print(np.shape(rate))
            rate = np.random.choice(np.ravel(rate), size=(num_samples, self.num_ues), replace=True)

        # compute G matrix. Since an approximate 25dB SINR value is recommended for medium- to high-quality data
        # networks, we could scale the diagonal entries in G by approximately 10 ** (22.5 / 10) * num_ues to compensate
        # for extra techniques like antenna gain and coding gain, etc.
        dist = np.zeros((self.num_ues, self.num_aps))
        for i in range(num_samples):
            for ue in range(self.num_ues):
                for ap in range(self.num_aps):
                    dist[ue, ap] = np.maximum(np.linalg.norm(ue_locs[i, ue] - ap_locs[i, ap], ord=1), 0.035)
            # compute path loss according to distances. The formula requires d>=0.035 to hold.
            path_loss = dist ** (-3.76)
            for j in range(self.num_aps):
                gain[i, j, :, :] = np.matmul(np.diag(path_loss[:, j]) ** 0.5, gain[i, j, :, :])
                gain[i, j, :, :] = np.matmul(gain[i, j, :, :], np.diag(path_loss[:, j]) ** 0.5)
                gain[i, j, :, :] += np.eye(self.num_ues) * gain[i, j, :, :] * 10 ** (25 / 10) * self.num_ues
        return [gain * 10 ** (-12.8), rate]


if __name__ == '__main__':
    # set global parameters
    num_aps = 4
    num_ues = 2
    noise_dbm = -60
    eps = 1.0e-06
    filename_gain = 'channel_gain.mat'
    filename_rate = 'ue_rates.mat'
    filename_locs = 'locs.mat'
    fading_type = 'weibull_gain'
    num_samples = 1000
    num_training = 900
    num_test = 100
    rand_generator = RandGenerator(num_ues, num_aps, filename_gain, filename_rate, filename_locs)
    gains, ue_rates = rand_generator.generate(num_samples, fading_type, offset=0)
