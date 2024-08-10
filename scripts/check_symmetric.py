import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from shapely.geometry import Polygon, Point
from helper import read_csv

def find_symmetry_via_fitted_lines(points):
    points = np.array(points, dtype=np.float32).reshape(-1, 2)

    # Fit y = mx + c
    x = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    model_y_on_x = LinearRegression()
    model_y_on_x.fit(x, y)
    slope_y_on_x = model_y_on_x.coef_[0]
    intercept_y_on_x = model_y_on_x.intercept_

    # Fit x = my + c
    model_x_on_y = LinearRegression()
    model_x_on_y.fit(y.reshape(-1, 1), x)
    slope_x_on_y = model_x_on_y.coef_[0]
    intercept_x_on_y = model_x_on_y.intercept_

    # Reflect points across y = mx + c
    def reflect_points_y_on_x(points, slope, intercept):
        reflected_points = []
        for x, y in points:
            perp_slope = -1 / slope
            perp_intercept = y - perp_slope * x
            inter_x = (perp_intercept - intercept) / (slope - perp_slope)
            inter_y = slope * inter_x + intercept
            ref_x = 2 * inter_x - x
            ref_y = 2 * inter_y - y
            reflected_points.append([ref_x, ref_y])
        return np.array(reflected_points)

    # Reflect points across x = my + c
    def reflect_points_x_on_y(points, slope, intercept):
        reflected_points = []
        for x, y in points:
            perp_slope = -1 / slope
            perp_intercept = x - perp_slope * y
            inter_y = (perp_intercept - intercept) / (slope - perp_slope)
            inter_x = slope * inter_y + intercept
            ref_y = 2 * inter_y - y
            ref_x = 2 * inter_x - x
            reflected_points.append([ref_x, ref_y])
        return np.array(reflected_points)

    reflected_points_y_on_x = reflect_points_y_on_x(points, slope_y_on_x, intercept_y_on_x)
    reflected_points_x_on_y = reflect_points_x_on_y(points, slope_x_on_y, intercept_x_on_y).squeeze(2)

    # Create polygons using the reflected points
    polygon_y_on_x = Polygon(reflected_points_y_on_x)
    polygon_x_on_y = Polygon(reflected_points_x_on_y)

    # Calculate mean distances
    distances_y_on_x = [polygon_y_on_x.exterior.distance(Point(p)) for p in points]
    distances_x_on_y = [polygon_x_on_y.exterior.distance(Point(p)) for p in points]

    mean_distance_y_on_x = np.mean(distances_y_on_x)
    mean_distance_x_on_y = np.mean(distances_x_on_y)

    print("Mean distance between points and their reflections across y = mx + c:", mean_distance_y_on_x)
    print("Mean distance between points and their reflections across x = my + c:", mean_distance_x_on_y)

    symmetry_threshold = 1.0  # Adjust as needed

    y_on_x_symmetry = mean_distance_y_on_x < symmetry_threshold
    x_on_y_symmetry = mean_distance_x_on_y < symmetry_threshold

    return y_on_x_symmetry, x_on_y_symmetry, (slope_y_on_x, intercept_y_on_x), (slope_x_on_y, intercept_x_on_y), reflected_points_y_on_x, reflected_points_x_on_y

# Example usage
# points = read_csv(r'problems\problems\vase.csv')[0][0]
# y_on_x_symmetry, x_on_y_symmetry, line_y_on_x, line_x_on_y, reflected_points_y_on_x, reflected_points_x_on_y = find_symmetry_via_fitted_lines(points)

# # Plotting
# plt.figure(figsize=(10, 10))
# plt.scatter(points[:, 0], points[:, 1], color='blue', label='Original Points', s=100)
# plt.scatter(reflected_points_y_on_x[:, 0], reflected_points_y_on_x[:, 1], color='orange', label='Reflected Points (y = mx + c)')
# plt.scatter(reflected_points_x_on_y[:, 0], reflected_points_x_on_y[:, 1], color='purple', label='Reflected Points (x = my + c)')

# if y_on_x_symmetry:
    # slope, intercept = line_y_on_x
    # x_vals = np.array(plt.gca().get_xlim())
    # y_vals = slope * x_vals + intercept
    # plt.plot(x_vals, y_vals, color='green', linestyle='--', label='Symmetry Line (y = mx + c)')

# if x_on_y_symmetry:
    # slope, intercept = line_x_on_y
    # y_vals = np.array(plt.gca().get_ylim())
    # x_vals = slope * y_vals + intercept
    # plt.plot(x_vals, y_vals, color='red', linestyle='--', label='Symmetry Line (x = my + c)')

# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.title('Shape with Symmetry Lines')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig('misc-outputs/vase-symmetry.png')
# plt.show()

# if y_on_x_symmetry:
#     print("The shape has symmetry along the line y = mx + c.")
#     print("Symmetry line (y = mx + c):", line_y_on_x)
# else:
#     print("The shape does not have symmetry along the line y = mx + c.")

# if x_on_y_symmetry:
#     print("The shape has symmetry along the line x = my + c.")
#     print("Symmetry line (x = my + c):", line_x_on_y)
# else:
#     print("The shape does not have symmetry along the line x = my + c.")
