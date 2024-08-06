import numpy as np
from shapely.geometry import Polygon, Point, MultiPoint
from helper_for_csvs import read_csv, plot
import cv2
from scipy.optimize import least_squares

def calculate_polygon_error(original_points, fitted_points):
    fitted_polygon = Polygon(fitted_points).exterior
    distances = [fitted_polygon.distance(Point(p)) for p in original_points]
    return np.mean(distances)

def fit_shapes(polyline_group):
    results = []
    names = []
    errors = []
    for polylines in polyline_group:
        for polyline in polylines:
            best_shape_points = fit_shape(polyline)
            results.append(best_shape_points[0])
            names.append(best_shape_points[2])
            errors.append(best_shape_points[1])
    return [results], names, errors

def fit_shape(points):
    # Ensure points is a numpy array
    points = np.array(points)
    
    # Fit different shapes and calculate errors
    line_error, line_points = fit_line(points)
    circle_error, circle_points = fit_circle(points)
    rectangle_error, rectangle_points = fit_rectangle(points)
    ellipse_error, ellipse_points = fit_ellipse(points)
    star_error, star_points = fit_star(points)
    triangle_error, triangle_points = fit_triangle(points)
    square_error, square_points = fit_square(points)
    pentagon_error, pentagon_points = fit_pentagon(points)
    hexagon_error, hexagon_points = fit_hexagon(points)
    
    # Collect all errors and their corresponding points
    errors = [
        ["line", line_error, line_points],
        ["circle", circle_error, circle_points],
        ["rectangle", rectangle_error, rectangle_points],
        ["ellipse", ellipse_error, ellipse_points],
        ["star", star_error, star_points],
        ["triangle", triangle_error, triangle_points],
        ["square", square_error, square_points],
        ["pentagon", pentagon_error, pentagon_points],
        ["hexagon", hexagon_error, hexagon_points]
    ]
    errors = sorted(errors, key=lambda x: x[1])
    best_shape = errors[0][0]
    lowest_error = errors[0][1]
    best_points = errors[0][2]

    print(f"Errors: line={line_error:.2f} circle={circle_error:.2f}, rectangle={rectangle_error:.9f}, ellipse={ellipse_error:.2f}, "
          f"star={star_error:.2f}, triangle={triangle_error:.2f}, square={square_error:.9f}, "
          f"pentagon={pentagon_error:.2f}, hexagon={hexagon_error:.2f}")
    
    return best_points, lowest_error, best_shape

def fit_line(points):
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    line = m * x + c
    line_points = np.column_stack((x, line))
    error = calculate_polygon_error(points, line_points)
    return error, line_points

def fit_circle(points):
    points = np.array(points)
    
    # Calculate initial guess for circle center and radius
    x_m = np.mean(points[:, 0])
    y_m = np.mean(points[:, 1])
    center_initial = np.array([x_m, y_m])
    
    def calc_R(c):
        """Calculate the distance of each data point from the center c (xc, yc)."""
        return np.sqrt((points[:, 0] - c[0])**2 + (points[:, 1] - c[1])**2)
    
    def objective_function(c):
        """Calculate the algebraic distance between the data points and the mean circle centered at c."""
        Ri = calc_R(c)
        return Ri - Ri.mean()
    
    center_optimized = least_squares(objective_function, center_initial).x
    radius_optimized = calc_R(center_optimized).mean()
    
    # Generate fitted circle points
    circle_points = np.array([
        [center_optimized[0] + radius_optimized * np.cos(theta), 
         center_optimized[1] + radius_optimized * np.sin(theta)]
        for theta in np.linspace(0, 2 * np.pi, 100)
    ])
    
    circle_error = calculate_polygon_error(points, circle_points)
    
    return circle_error, circle_points

def fit_rectangle(points):
    multi_point = MultiPoint(points)
    rect = multi_point.minimum_rotated_rectangle
    rect_points = np.array(rect.exterior.coords)
    
    rectangle_error = calculate_polygon_error(points, rect_points)
    
    return rectangle_error, rect_points

def fit_ellipse(points):
    if len(points) < 5:
        return float('inf'), points
    points = np.array(points, dtype=np.float32)
    ellipse = cv2.fitEllipse(points)
    ellipse_points = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0]//2), int(ellipse[1][1]//2)), int(ellipse[2]), 0, 360, 1)
    ellipse_points = np.array(ellipse_points, dtype=np.float32)
    
    ellipse_error = calculate_polygon_error(points, ellipse_points)
    
    return ellipse_error, ellipse_points

def fit_star(points):
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    
    sorted_indices = np.argsort(angles)
    sorted_distances = distances[sorted_indices]
    
    peaks = (sorted_distances > np.roll(sorted_distances, 1)) & (sorted_distances > np.roll(sorted_distances, -1))
    valleys = (sorted_distances < np.roll(sorted_distances, 1)) & (sorted_distances < np.roll(sorted_distances, -1))
    
    num_peaks = np.sum(peaks)
    num_valleys = np.sum(valleys)
    
    if num_peaks < 3 or num_valleys < 3 or abs(num_peaks - num_valleys) > 1:
        return float('inf'), points
    
    peak_distances = sorted_distances[peaks]
    valley_distances = sorted_distances[valleys]
    
    if np.mean(peak_distances) <= 1.5 * np.mean(valley_distances):
        return float('inf'), points
    
    n_points = num_peaks * 2
    star_points = []
    for i in range(n_points):
        angle = i * 2 * np.pi / n_points
        radius = np.mean(peak_distances) if i % 2 == 0 else np.mean(valley_distances)
        x = centroid[0] + radius * np.cos(angle)
        y = centroid[1] + radius * np.sin(angle)
        star_points.append((x, y))
    
    star_points = np.array(star_points)
    star_error = calculate_polygon_error(points, star_points)
    star_points = np.append(star_points, [star_points[0]], axis=0)
    return star_error, star_points

def fit_triangle(points):
    hull = cv2.convexHull(np.array(points, dtype=np.float32))
    if len(hull) > 3:
        hull = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    
    if len(hull) != 3:
        return float('inf'), points
    
    triangle_points = np.array(hull).reshape(-1, 2)
    triangle_error = calculate_polygon_error(points, triangle_points)
    
    return triangle_error, triangle_points

def fit_square(points):
    rect = cv2.minAreaRect(np.array(points, dtype=np.float32))
    box = cv2.boxPoints(rect)
    
    edges = np.roll(box, 1, axis=0) - box
    edge_lengths = np.linalg.norm(edges, axis=1)
    if max(edge_lengths) / min(edge_lengths) > 1.2:
        return float('inf'), points
    
    square_points = np.array(box)
    square_error = calculate_polygon_error(points, square_points)
    
    return square_error, square_points

def fit_regular_polygon(points, n_sides):
    hull = cv2.convexHull(np.array(points, dtype=np.float32))
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    
    if len(approx) != n_sides:
        return float('inf'), points
    
    polygon_points = np.array(approx).reshape(-1, 2)
    polygon_error = calculate_polygon_error(points, polygon_points)
    
    return polygon_error, polygon_points

def fit_pentagon(points):
    return fit_regular_polygon(points, 5)

def fit_hexagon(points):
    return fit_regular_polygon(points, 6)

# Read polylines from CSV
filename = 'problems\problems\isolated.csv'
polylines = read_csv(filename)

# Fit shapes to polylines and get best fit shapes
shapes, names, errors = fit_shapes(polylines)

# Plot the best fit shapes
plot(shapes, filename.split('\\')[-1].replace('.csv', '.png'), names)
