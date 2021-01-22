from cnbc.cnbc import PointClass
import matplotlib.pyplot as plt


def plot_cnbc_clusters(x, y, clusters):
    # plot cnbc output
    classes = clusters.replace({PointClass.NOISE: PointClass.NOISE.value})
    unique = list(set(classes))
    colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
    for i, u in enumerate(unique):
        xi = [x[j] for j in range(len(x)) if classes[j] == u]
        yi = [y[j] for j in range(len(x)) if classes[j] == u]
        label = "NOISE" if u == 0 else str(u)
        plt.scatter(xi, yi, color=colors[i], label=label)
    plt.legend()
