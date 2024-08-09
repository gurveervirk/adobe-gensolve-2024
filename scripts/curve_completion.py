import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, distance
from scipy.interpolate import make_interp_spline
from helper_for_csvs import read_csv

def interpolate_missing_parts(polyline):
    # Convex Hull to get boundary points
    hull = ConvexHull(polyline)
    hull_points = polyline[hull.vertices]
    hull_points = np.roll(hull_points, -1, axis=0)  # Roll the points to get a continuous curve (one end to another)

    # Fit B-spline to the convex hull points
    t = np.linspace(0, 1, len(hull_points))
    spl = make_interp_spline(t, hull_points, k=3)  # k=3 for cubic spline
    t_new = np.linspace(0, 2, 600)
    threshold = 1  # Distance threshold to stop the process
    interpolated_points = []
    flag = False  # Initialize flag

    # Iteratively generate new points and check distance
    for t_val in t_new:
        new_point = spl(t_val)
        min_dist = np.min([distance.euclidean(new_point, point) for point in hull_points])
        interpolated_points.append(new_point)
        if min_dist > threshold:
            flag = True  # Set flag to True when new point is outside original hull_points
        
        if flag and min_dist < threshold:
            break 
    
    return np.array(interpolated_points)

# Load polyline data
path = r'problems\problems\occlusion2.csv'
polylines = read_csv(path)
curve_to_complete = polylines[4][0]

name = path.split("\\")[-1][:-4]

# Interpolate missing parts using convex hull and B-spline
completed_polyline = interpolate_missing_parts(curve_to_complete)

# Plot the original polyline and the completed polyline
fig, ax = plt.subplots()
ax.plot(curve_to_complete[:, 0], curve_to_complete[:, 1], 'g-', label='Original Polyline', linewidth=4)
ax.plot(completed_polyline[:, 0], completed_polyline[:, 1], 'purple', label='Completed Polyline', linewidth=2)
ax.legend()
plt.savefig('misc-outputs/' + f'curve_completion_{name}.png')
plt.show()
