import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.interpolate import interp1d
from helper import read_csv

def extrapolate_missing_parts(polyline):
    # Fit a quadratic interpolation to the polyline
    t = np.linspace(0, 1, len(polyline))
    interp_func = interp1d(t, polyline, kind='quadratic', axis=0, fill_value="extrapolate")
    t_new = np.linspace(0, 1.05, 600)
    threshold = 1  # Distance threshold to stop the process
    interpolated_points = []
    flag = False  # Initialize flag
    completed = False  # Initialize completed flag

    # Iteratively generate new points and check distance
    for t_val in t_new:
        new_point = interp_func(t_val)
        min_dist = np.min([distance.euclidean(new_point, point) for point in polyline])
        interpolated_points.append(new_point)
        if min_dist > threshold:
            flag = True  # Set flag to True when new point is outside original polyline
        
        if flag and min_dist < threshold:
            completed = True
            break 
    
    return np.array(interpolated_points), completed

# Sample usage
# path = r'problems\problems\occlusion1.csv'
# polylines = read_csv(path)
# curve_to_complete = polylines[0][0]

# name = path.split("\\")[-1][:-4]

# # Interpolate missing parts using convex hull and B-spline
# completed_polyline, _ = extrapolate_missing_parts(curve_to_complete)

# # Plot the original polyline and the completed polyline
# fig, ax = plt.subplots()
# ax.plot(curve_to_complete[:, 0], curve_to_complete[:, 1], 'g-', label='Original Polyline', linewidth=4)
# ax.plot(completed_polyline[:, 0], completed_polyline[:, 1], 'purple', label='Completed Polyline', linewidth=2)
# ax.legend()
# # plt.savefig('misc-outputs/' + f'curve_completion_{name}.png')
# plt.show()
