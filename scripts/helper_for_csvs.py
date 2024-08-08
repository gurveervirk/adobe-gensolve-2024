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

def plot(paths_XYs, filename, names, symmetries):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    ind = 0
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, XYs in enumerate(paths_XYs):
        for j, XY in enumerate(XYs):
            c = colours[ind % len(colours)]
            ind += 1
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
            midpoint = np.mean(XY, axis=0)
            ax.text(midpoint[0], midpoint[1], names[j], fontsize=12, color=c, ha='center')
            
            # Plot symmetry lines within the plot limits
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            
            for slope, intercept in symmetries[j]:
                if np.isinf(slope):  # Vertical line
                    x_vals = np.array([intercept, intercept])
                    y_vals = np.array([y_min, y_max])
                elif slope == 0:  # Horizontal line
                    x_vals = np.array([x_min, x_max])
                    y_vals = np.array([intercept, intercept])
                else:  # Non-vertical line
                    x_vals = np.array([x_min, x_max])
                    y_vals = slope * x_vals + intercept
                    # Clipping y_vals to plot limits
                    y_vals = np.clip(y_vals, y_min, y_max)
                    # Recalculating x_vals to fit within the clipped y_vals
                    x_vals = (y_vals - intercept) / slope
                    x_vals = np.clip(x_vals, x_min, x_max)
                
                ax.plot(x_vals, y_vals, c + '--', linewidth=1)
        
    ax.set_aspect('equal')
    plt.savefig(f'misc-outputs/{filename}')
    plt.show()

def plot_simple(paths_XYs, filename):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for j, XY in enumerate(XYs):
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
            ax.set_aspect('equal')
            average_XY = np.mean(XY, axis=0)
            ax.text(average_XY[0], average_XY[1], str(i) + ' ' + str(j), fontsize=12, color=c, ha='center')
    plt.savefig(f'misc-outputs/{filename}')
    plt.show()