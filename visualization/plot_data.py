import matplotlib.pyplot as plt


def plot_data(data_list, colors=None, labels=None):
    plt.figure(figsize=(10, 6))

    if colors is None:
        colors = [None] * len(data_list)
    if labels is None:
        labels = [None] * len(data_list)

    for (x, y), color, label in zip(data_list, colors, labels):
        plt.plot(x, y, color=color, label=label)

    plt.grid(True)
    plt.xlabel("Cl")
    plt.ylabel("alpha")
    plt.title("Cl(alpha)", fontweight='bold')

    if any(labels):
        plt.legend()

    plt.show()