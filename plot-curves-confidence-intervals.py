import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    # plotting curves with confidence intervals
    pm = np.load('pm.npy')
    (num_aps, num_samples, num_iterations) = np.shape(pm)
    q = np.zeros(shape=(num_aps * num_samples * num_iterations, 3))
    cnt = 0
    for i in range(num_aps):
        for j in range(num_samples):
            for k in range(num_iterations):
                q[cnt, 0] = k + 1
                q[cnt, 1] = np.log(pm[i, j, k])
                q[cnt, 2] = i + 1
                cnt += 1
    df = pd.DataFrame(q, columns=['iterations', 'power (dBm)', 'access point IDs'])
    sns.set_theme(style="darkgrid")
    sns.lineplot(data=df, x='iterations', y='power (dBm)', hue='access point IDs', ci=68)
    plt.legend(labels=['ap1', 'ap2', 'ap3', 'ap4'])
    plt.show()
