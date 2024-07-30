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

def plot(paths_XYs):
    fig, ax = plt.subplots(tight_layout = True, figsize=(8, 8))
    for i , XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs :
            ax.plot(XY[: , 0] , XY [: , 1] , c=c , linewidth =2)
            ax.set_aspect('equal')
    plt.show()
