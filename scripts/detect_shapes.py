import numpy as np
from shapely.geometry import Polygon, Point
from shapely.affinity import rotate, scale, translate
from helper_for_csvs import read_csv, plot

def fit_shapes(polyline_group):
    results = []
    for polylines in polyline_group:
        for polyline in polylines:
            best_shape_points = fit_shape(polyline)
            results.append(best_shape_points)
    return [results]

def fit_shape(points):
    # Ensure points is a numpy array
    points = np.array(points)
    
    # Fit different shapes and calculate errors
    line_error, line_points = fit_line(points)
    circle_error, circle_points = fit_circle(points)
    rectangle_error, is_rectangular, rectangle_points = fit_rectangle(points)
    ellipse_error, ellipse_points = fit_ellipse(points)
    star_error, star_points = fit_star(points)
    
    # Collect all errors and their corresponding points
    errors = [
        ["straight line", line_error, line_points],
        ["circle", circle_error, circle_points],
        ["rectangle", rectangle_error, rectangle_points],
        ["ellipse", ellipse_error, ellipse_points],
        ["star", star_error, star_points]
    ]
    errors = sorted(errors, key=lambda x: x[1])
    best_shape = errors[0][0]
    best_points = errors[0][2]
    
    # If the best shape is not rectangle but it's approximately rectangular
    if best_shape != 'rectangle' and is_rectangular:
        best_shape = 'rectangle'
        best_points = rectangle_points
    
    return best_points

def fit_line(points):
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    line = m * x + c
    line_points = np.column_stack((x, line))
    error = np.mean((y - line) ** 2)
    return error, line_points

def fit_circle(points):
    poly = Polygon(points)
    center = poly.centroid
    radii = [Point(p).distance(center) for p in points]
    radius = np.mean(radii)
    circle = Point(center).buffer(radius)
    circle_points = np.array(circle.exterior.coords)
    error = abs(poly.area - circle.area)
    return error, circle_points

def fit_rectangle(points):
    poly = Polygon(points)
    min_rect = poly.minimum_rotated_rectangle
    rectangle_points = np.array(min_rect.exterior.coords)
    
    # Calculate area difference
    area_diff = abs(poly.area - min_rect.area)

    # Calculate angles for all vertices in the original polygon
    angles = []
    n = len(points)
    count_greater_than_one = 0
    for i in range(n):
        v1 = points[i] - points[i-1]
        v2 = points[(i+1) % n] - points[i]
        angle = np.arctan2(np.cross(v2, v1), np.dot(v2, v1))
        angle = abs(angle)
        angles.append(angle)
        if angle > 1:
            count_greater_than_one += 1
    
    is_rectangular = count_greater_than_one == 4
    
    return area_diff, is_rectangular, rectangle_points

def fit_ellipse(points):
    x = points[:, 0]
    y = points[:, 1]
    x_mean, y_mean = np.mean(x), np.mean(y)
    
    # Calculate the covariance matrix
    cov = np.cov(x - x_mean, y - y_mean)
    
    # Calculate the eigenvalues and eigenvectors
    evals, evecs = np.linalg.eig(cov)
    
    # Sort eigenvalues in descending order
    sort_indices = np.argsort(evals)[::-1]
    evals = evals[sort_indices]
    evecs = evecs[:, sort_indices]
    
    # Calculate the angle of rotation
    angle = np.arctan2(evecs[1, 0], evecs[0, 0])
    
    # Calculate the semi-major and semi-minor axes
    a = np.sqrt(evals[0]) * 2
    b = np.sqrt(evals[1]) * 2
    
    # Create an ellipse using shapely
    circle = Point(0, 0).buffer(1)
    ellipse = scale(circle, a, b)
    ellipse = rotate(ellipse, angle, origin='center')
    ellipse = translate(ellipse, x_mean, y_mean)
    ellipse_points = np.array(ellipse.exterior.coords)
    
    # Calculate the error
    poly = Polygon(points)
    error = abs(poly.area - ellipse.area)
    
    return error, ellipse_points

def fit_star(points):
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    
    # Sort points by angle
    sorted_indices = np.argsort(angles)
    sorted_distances = distances[sorted_indices]
    
    # Find local maxima (peaks) and minima (valleys)
    peaks = (sorted_distances > np.roll(sorted_distances, 1)) & (sorted_distances > np.roll(sorted_distances, -1))
    valleys = (sorted_distances < np.roll(sorted_distances, 1)) & (sorted_distances < np.roll(sorted_distances, -1))
    
    num_peaks = np.sum(peaks)
    num_valleys = np.sum(valleys)
    
    # Check if number of peaks and valleys are similar and > 2 (for at least a 5-pointed star)
    if num_peaks < 3 or num_valleys < 3 or abs(num_peaks - num_valleys) > 1:
        return float('inf'), points
    
    peak_distances = sorted_distances[peaks]
    valley_distances = sorted_distances[valleys]
    
    # Check if peaks are significantly larger than valleys
    if np.mean(peak_distances) <= 1.5 * np.mean(valley_distances):
        return float('inf'), points
    
    # Create a star shape
    n_points = num_peaks * 2
    star_points = []
    for i in range(n_points):
        angle = i * 2 * np.pi / n_points
        radius = np.mean(peak_distances) if i % 2 == 0 else np.mean(valley_distances)
        x = centroid[0] + radius * np.cos(angle)
        y = centroid[1] + radius * np.sin(angle)
        star_points.append((x, y))
    
    star = Polygon(star_points)
    star_points = np.array(star.exterior.coords)
    poly = Polygon(points)
    error = abs(poly.area - star.area)
    
    return error, star_points

# Read polylines from CSV
polylines = read_csv(r'C:\Users\GURDARSH VIRK\OneDrive\Documents\adobe-gensolve-2024\problems\problems\isolated.csv')

# Fit shapes to polylines and get best fit shapes
shapes = fit_shapes(polylines)
# Plot the best fit shapes
plot(shapes)
