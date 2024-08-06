import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import LineString
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from helper_for_csvs import read_csv, plot

# Example polylines (replace with actual data)
polylines = read_csv(r'problems\problems\frag0.csv')

def merge_polylines(polylines, epsilon=1e-2):
    points = np.vstack(polylines)
    clustering = DBSCAN(eps=epsilon, min_samples=1).fit(points)
    merged_polylines = []
    for cluster_id in np.unique(clustering.labels_):
        cluster_points = points[clustering.labels_ == cluster_id]
        merged_polylines.append(cluster_points)
    return merged_polylines

def simplify_polylines(polylines, tolerance=1.0):
    simplified_polylines = []
    for polyline in polylines:
        line = LineString(polyline)
        simplified_line = line.simplify(tolerance)
        simplified_polylines.append(np.array(simplified_line.coords))
    return simplified_polylines

def fit_bezier_curve(polyline, degree=3):
    tck, u = splprep(polyline.T, k=degree, s=0)
    new_points = splev(np.linspace(0, 1, 100), tck)
    return np.vstack(new_points).T

merged_polylines = merge_polylines(polylines)
simplified_polylines = simplify_polylines(merged_polylines)
bezier_curves = [fit_bezier_curve(polyline) for polyline in simplified_polylines]

# Plotting the results
plt.figure(figsize=(10, 6))
for curve in bezier_curves:
    plt.plot(curve[:, 0], curve[:, 1], label='Bezier Curve')
plt.legend()
plt.show()
