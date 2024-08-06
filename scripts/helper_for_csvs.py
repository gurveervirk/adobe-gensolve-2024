import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

colours = list(mcolors.TABLEAU_COLORS)

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path,delimiter =',')
    path_XYs = []
    for i in np . unique ( np_path_XYs [: , 0]):
        npXYs = np_path_XYs [ np_path_XYs [: , 0] == i ][: , 1:]
        XYs = []
        for j in np . unique ( npXYs [: , 0]):
            XY = npXYs [ npXYs [: , 0] == j ][: , 1:]
            XYs . append ( XY )
        path_XYs . append ( XYs )
    return path_XYs

def plot(paths_XYs, filename, names):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    ind = 0
    for i, XYs in enumerate(paths_XYs):
        for j, XY in enumerate(XYs):
            c = colours[ind % len(colours)]
            ind += 1
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
            midpoint = np.mean(XY, axis=0)
            ax.text(midpoint[0], midpoint[1], names[j], fontsize=12, color=c, ha='center')
        ax.set_aspect('equal')
    plt.savefig(f'misc-outputs/{filename}')
    plt.show()