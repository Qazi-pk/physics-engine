import matplotlib.pyplot as plt
import numpy as np


def plot_kepler(r, T):
    plt.scatter(r, T)

    plt.xlabel("Orbital Radius (r)")
    plt.ylabel("Orbital Period (T)")

    plt.title("Kepler Law Discovery")

    plt.show()


def plot_loglog(r, T):
    plt.scatter(np.log(r), np.log(T))

    plt.xlabel("log(r)")
    plt.ylabel("log(T)")

    plt.title("Log-Log plot (slope ≈ 1.5)")

    plt.show()
