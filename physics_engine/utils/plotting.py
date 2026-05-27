import matplotlib.pyplot as plt


def plot_series(x, y, *, title="", xlabel="x", ylabel="y"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig, ax
