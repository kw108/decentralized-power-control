import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    # plotting curves with confident intervals
    pm = np.load('pm.npy')
    q = np.zeros(shape=(400, 3))
    for i in range(100):
        q[i, 0] = i % 10 + 1
        q[i, 1] = pm[0, i // 10, i % 10]
        q[i, 2] = 0
    for i in range(100, 200):
        q[i, 0] = i % 10 + 1
        q[i, 1] = pm[1, (i - 100) // 10, i % 10]
        q[i, 2] = 1
    for i in range(200, 300):
        q[i, 0] = i % 10 + 1
        q[i, 1] = pm[2, (i - 200) // 10, i % 10]
        q[i, 2] = 2
    for i in range(300, 400):
        q[i, 0] = i % 10 + 1
        q[i, 1] = pm[3, (i - 300) // 10, i % 10]
        q[i, 2] = 3
    df = pd.DataFrame(q, columns=['iter', 'power', 'APs'])
    sns.set_theme(style="darkgrid")
    sns.lineplot(data=df, x='iter', y='power', hue='APs')
