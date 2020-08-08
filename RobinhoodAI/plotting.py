import numpy as np
import matplotlib.pyplot as plt
import os.path

from .train import max_forecast

plt.style.use('./etc/rh_plot.mplstyle')


def plot(y_obs, y_pred, filename='plot'):
    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    fig, ax = plt.subplots()

    X_obs = np.arange(-y_obs.shape[0] + 1, 1)
    X_pred = np.arange(1, max_forecast + 1)

    ms = 16
    lw = 0.6
    ax.scatter(X_obs, y_obs, 'o', color='k', marker='o', s=ms, edgecolor='k',
               linewidths=lw)
    ax.scatter(X_pred, y_pred, 'o', color='r', marker='o', s=ms, edgecolor='k',
               linewidths=lw)

    ax.set_xlabel('day relative to present')
    ax.set_ylabel('high price')

    fig.tight_layout()

    fig.savefig(f'{filename}.png', format='png')

    plt.close('all')
