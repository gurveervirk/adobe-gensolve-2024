import numpy as np
from shapely.geometry import Polygon, Point, MultiPoint
from scipy.optimize import least_squares
from helper import read_csv, plot
import cv2
from scipy.interpolate import splev, splrep
import scipy
from sklearn.linear_model import LinearRegression
from find_reflection_symmetry_line import find_reflection_symmetry_parallel, find_centroid

def points_are_close(p1, p2, tol=1e-5):
    distance = np.linalg.norm(np.array(p1) - np.array(p2))
    return distance < tol

def calculate_polygon_error(original_points, fitted_points):
    fitted_polygon = Polygon(fitted_points).exterior
    distances = [fitted_polygon.distance(Point(p)) for p in original_points]
    return np.mean(distances)

def fit_shapes(polyline_group):
    results = []
    names = []
    errors = []
    symmetries = []
    for polylines in polyline_group:
        for polyline in polylines:
            best_shape_points, lowest_error, best_shape, symmetry_lines = fit_shape(polyline)
            if lowest_error < 10:
                results.append(best_shape_points)
                names.append(best_shape)
                errors.append(lowest_error)
                symmetries.append(symmetry_lines)
            else:
                bezier_error, bezier_points, symmetry_lines = fit_bezier_curve(polyline)
                if bezier_error < 10:
                    results.append(bezier_points)
                    names.append("bezier")
                    errors.append(bezier_error)
                    symmetries.append(symmetry_lines)
                else:
                    spline_error, spline_points, symmetry_lines = fit_b_spline(polyline)
                    results.append(spline_points)
                    names.append("b-spline")
                    errors.append(spline_error)
                    symmetries.append(symmetry_lines)
    return [results], names, symmetries

def fit_shape(points):
    # Ensure points is a numpy array
    points = np.array(points)

    # Check if the points are already closed and are symmetric
    if np.all(points[0] == points[-1]) or points_are_close(points[0], points[-1]):
        # Check if the points are symmetric
        centroid = find_centroid(points)
        symmetry_line, error = find_reflection_symmetry_parallel(points, centroid)
        if error < 1:
            return points, error, "closed", [symmetry_line]
    
    # Fit different shapes and calculate errors
    line_error, line_points, line_symmetry = fit_line(points)
    circle_error, circle_points, circle_symmetry = fit_circle(points)
    rectangle_error, rectangle_points, rectangle_symmetry = fit_rectangle(points)
    ellipse_error, ellipse_points, ellipse_symmetry = fit_ellipse(points)
    star_error, star_points, star_symmetry = fit_star(points)
    triangle_error, triangle_points, triangle_symmetry = fit_triangle(points)
    square_error, square_points, square_symmetry = fit_square(points)
    pentagon_error, pentagon_points, pentagon_symmetry = fit_pentagon(points)
    hexagon_error, hexagon_points, hexagon_symmetry = fit_hexagon(points)
    
    # Collect all errors and their corresponding points
    errors = [
        ["line", line_error, line_points, line_symmetry],
        ["circle", circle_error, circle_points, circle_symmetry],
        ["rectangle", rectangle_error, rectangle_points, rectangle_symmetry],
        ["ellipse", ellipse_error, ellipse_points, ellipse_symmetry],
        ["star", star_error, star_points, star_symmetry],
        ["triangle", triangle_error, triangle_points, triangle_symmetry],
        ["square", square_error, square_points, square_symmetry],
        ["pentagon", pentagon_error, pentagon_points, pentagon_symmetry],
        ["hexagon", hexagon_error, hexagon_points, hexagon_symmetry]
    ]
    errors = sorted(errors, key=lambda x: x[1])
    best_shape = errors[0][0]
    lowest_error = errors[0][1]
    best_points = errors[0][2]
    symmetry_lines = errors[0][3]

    print(f"Errors: line={line_error:.2f} circle={circle_error:.2f}, rectangle={rectangle_error:.9f}, ellipse={ellipse_error:.2f}, "
          f"star={star_error:.2f}, triangle={triangle_error:.2f}, square={square_error:.9f}, "
          f"pentagon={pentagon_error:.2f}, hexagon={hexagon_error:.2f}")
    
    return best_points, lowest_error, best_shape, symmetry_lines

def fit_line(points):
    x = points[:, 0]
    y = points[:, 1]
    
    # Fit y on x
    X_y_on_x = x.reshape(-1, 1)
    model_y_on_x = LinearRegression().fit(X_y_on_x, y)
    m_y_on_x = model_y_on_x.coef_[0]
    c_y_on_x = model_y_on_x.intercept_
    line_y_on_x = m_y_on_x * x + c_y_on_x
    line_points_y_on_x = np.column_stack((x, line_y_on_x))
    error_y_on_x = calculate_polygon_error(points, line_points_y_on_x)
    
    # Fit x on y
    X_x_on_y = y.reshape(-1, 1)
    model_x_on_y = LinearRegression().fit(X_x_on_y, x)
    m_x_on_y = model_x_on_y.coef_[0]
    c_x_on_y = model_x_on_y.intercept_
    line_x_on_y = (x - c_x_on_y) / m_x_on_y
    line_points_x_on_y = np.column_stack((line_x_on_y, y))
    error_x_on_y = calculate_polygon_error(points, line_points_x_on_y)
    
    # Choose the best fit
    if error_y_on_x < error_x_on_y:
        best_error = error_y_on_x
        best_line_points = line_points_y_on_x
        best_symmetry_lines = [(m_y_on_x, c_y_on_x)]
    else:
        best_error = error_x_on_y
        best_line_points = line_points_x_on_y
        best_symmetry_lines = [(1 / m_x_on_y, -c_x_on_y / m_x_on_y)]
    
    return best_error, best_line_points, best_symmetry_lines

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
        for theta in np.linspace(0, 2 * np.pi, 1000)
    ])
    
    circle_error = calculate_polygon_error(points, circle_points)

    # Calculate symmetry lines
    multi_point = MultiPoint(circle_points)
    rect = multi_point.minimum_rotated_rectangle
    rect_points = np.array(rect.exterior.coords)

    symmetry_lines = []
    for i in range(2):
        p1 = rect_points[i]
        p2 = rect_points[(i + 1) % len(rect_points)]
        p3 = rect_points[(i + 2) % len(rect_points)]
        p4 = rect_points[(i + 3) % len(rect_points)]
        midpoint1 = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
        midpoint2 = [(p3[0] + p4[0]) / 2, (p3[1] + p4[1]) / 2]
        if abs(midpoint1[0] - midpoint2[0]) < 1e-6:
            slope = np.inf
            intercept = midpoint1[0]
        else:
            slope = (midpoint2[1] - midpoint1[1]) / (midpoint2[0] - midpoint1[0])
            intercept = midpoint1[1] - slope * midpoint1[0]
        symmetry_lines.append((slope, intercept))

        midpoint1 = [(p1[0] + p3[0]) / 2, (p1[1] + p3[1]) / 2]
        midpoint2 = [(p2[0] + p4[0]) / 2, (p2[1] + p4[1]) / 2]
        if abs(midpoint1[0] - midpoint2[0]) < 1e-6:
            slope = np.inf
            intercept = midpoint1[0]
        else:
            slope = (midpoint2[1] - midpoint1[1]) / (midpoint2[0] - midpoint1[0])
            intercept = midpoint1[1] - slope * midpoint1[0]
        symmetry_lines.append((slope, intercept))
    
    return circle_error, circle_points, symmetry_lines

def fit_rectangle(points):
    multi_point = MultiPoint(points)
    rect = multi_point.minimum_rotated_rectangle
    rect_points = np.array(rect.exterior.coords)
    
    rectangle_error = calculate_polygon_error(points, rect_points)

    # Calculate symmetry lines
    symmetry_lines = []
    for i in range(2):
        p1 = rect_points[i]
        p2 = rect_points[(i + 1) % len(rect_points)]
        p3 = rect_points[(i + 2) % len(rect_points)]
        p4 = rect_points[(i + 3) % len(rect_points)]
        midpoint1 = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
        midpoint2 = [(p3[0] + p4[0]) / 2, (p3[1] + p4[1]) / 2]
        if abs(midpoint1[0] - midpoint2[0]) < 1e-6:
            slope = np.inf
            intercept = midpoint1[0]
        else:
            slope = (midpoint2[1] - midpoint1[1]) / (midpoint2[0] - midpoint1[0])
            intercept = midpoint1[1] - slope * midpoint1[0]
        symmetry_lines.append((slope, intercept))

    return rectangle_error, rect_points, symmetry_lines

def fit_ellipse(points):
    if len(points) < 5:
        return float('inf'), points, []

    points = np.array(points, dtype=np.float32)
    ellipse = cv2.fitEllipse(points)
    center = ellipse[0]
    axes = ellipse[1]
    angle = ellipse[2]

    # ellipse_points = cv2.ellipse2Poly((int(center[0]), int(center[1])), (int(axes[0] // 2), int(axes[1] // 2)), int(angle), 0, 360, 1)
    # Convert angle from OpenCV to radians
    angle_rad = np.deg2rad(angle)

    # Semi-major and semi-minor axes
    a = axes[0] / 2
    b = axes[1] / 2

    # Generate more points on the ellipse using parametric equations
    num_points = 1000  # Increase this number for more points
    t = np.linspace(0, 2 * np.pi, num_points)
    ellipse_points = np.zeros((num_points, 2), dtype=np.float32)
    
    for i in range(num_points):
        x = a * np.cos(t[i])
        y = b * np.sin(t[i])
        
        # Rotate the points
        x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
        
        # Translate to the center
        ellipse_points[i] = [center[0] + x_rot, center[1] + y_rot]
    
    ellipse_points = np.array(ellipse_points, dtype=np.float32)
    
    ellipse_error = calculate_polygon_error(points, ellipse_points)
    
    # Calculate symmetry lines
    multi_point = MultiPoint(ellipse_points)
    rect = multi_point.minimum_rotated_rectangle
    rect_points = np.array(rect.exterior.coords)

    symmetry_lines = []
    for i in range(2):
        p1 = rect_points[i]
        p2 = rect_points[(i + 1) % len(rect_points)]
        p3 = rect_points[(i + 2) % len(rect_points)]
        p4 = rect_points[(i + 3) % len(rect_points)]
        midpoint1 = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
        midpoint2 = [(p3[0] + p4[0]) / 2, (p3[1] + p4[1]) / 2]
        if abs(midpoint1[0] - midpoint2[0]) < 1e-6:
            slope = np.inf
            intercept = midpoint1[0]
        else:
            slope = (midpoint2[1] - midpoint1[1]) / (midpoint2[0] - midpoint1[0])
            intercept = midpoint1[1] - slope * midpoint1[0]
        symmetry_lines.append((slope, intercept))
    
    return ellipse_error, ellipse_points, symmetry_lines

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
        return float('inf'), points, []
    
    peak_distances = sorted_distances[peaks]
    valley_distances = sorted_distances[valleys]
    
    if np.mean(peak_distances) <= 1.5 * np.mean(valley_distances):
        return float('inf'), points, []
    
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
    # Calculate symmetry lines
    symmetry_lines = []
    for i in range(num_peaks):
        p1 = star_points[i]
        p2 = centroid
        if p2[0] == p1[0]:
            slope = np.inf
            intercept = p1[0]
        else:
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
            intercept = p1[1] - slope * p1[0]
        symmetry_lines.append((slope, intercept))
    
    return star_error, star_points, symmetry_lines

def fit_triangle(points):
    hull = cv2.convexHull(np.array(points, dtype=np.float32))
    if len(hull) != 3:
        return float('inf'), points, []
    triangle_points = hull[:, 0, :]
    triangle_error = calculate_polygon_error(points, triangle_points)
    symmetry_lines = get_regular_polygon_symmetry(triangle_points, 3)
    return triangle_error, triangle_points, symmetry_lines

def fit_square(points):
    multi_point = MultiPoint(points)
    min_rectangle = multi_point.minimum_rotated_rectangle
    min_rectangle_points = np.array(min_rectangle.exterior.coords)
    
    edge_lengths = np.linalg.norm(np.diff(min_rectangle_points, axis=0), axis=1)
    if not np.allclose(edge_lengths, edge_lengths[0], rtol=0.1):
        return float('inf'), points, []
    
    square_points = min_rectangle_points[:4]
    square_error = calculate_polygon_error(points, square_points)

    # Calculate symmetry lines
    symmetry_lines = []
    for i in range(2):
        p1 = square_points[i]
        p2 = square_points[(i + 1) % len(square_points)]
        p3 = square_points[(i + 2) % len(square_points)]
        p4 = square_points[(i + 3) % len(square_points)]
        midpoint1 = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
        midpoint2 = [(p3[0] + p4[0]) / 2, (p3[1] + p4[1]) / 2]
        if abs(midpoint1[0] - midpoint2[0]) < 1e-6:
            slope = np.inf
            intercept = midpoint1[0]
        else:
            slope = (midpoint2[1] - midpoint1[1]) / (midpoint2[0] - midpoint1[0])
            intercept = midpoint1[1] - slope * midpoint1[0]
        symmetry_lines.append((slope, intercept))

        midpoint1 = [(p1[0] + p3[0]) / 2, (p1[1] + p3[1]) / 2]
        midpoint2 = [(p2[0] + p4[0]) / 2, (p2[1] + p4[1]) / 2]
        if abs(midpoint1[0] - midpoint2[0]) < 1e-6:
            slope = np.inf
            intercept = midpoint1[0]
        else:
            slope = (midpoint2[1] - midpoint1[1]) / (midpoint2[0] - midpoint1[0])
            intercept = midpoint1[1] - slope * midpoint1[0]
        symmetry_lines.append((slope, intercept))
    
    return square_error, square_points, symmetry_lines

def fit_pentagon(points):
    hull = cv2.convexHull(np.array(points, dtype=np.float32))
    if len(hull) != 5:
        return float('inf'), points, []
    pentagon_points = hull[:, 0, :]
    pentagon_error = calculate_polygon_error(points, pentagon_points)
    symmetry_lines = get_regular_polygon_symmetry(pentagon_points, 5)
    return pentagon_error, pentagon_points, symmetry_lines

def fit_hexagon(points):
    hull = cv2.convexHull(np.array(points, dtype=np.float32))
    if len(hull) != 6:
        return float('inf'), points, []
    hexagon_points = hull[:, 0, :]
    hexagon_error = calculate_polygon_error(points, hexagon_points)
    symmetry_lines = get_regular_polygon_symmetry(hexagon_points, 6)
    return hexagon_error, hexagon_points, symmetry_lines

def calculate_orientation(points):
    # Calculate the covariance matrix of the points
    cov_matrix = np.cov(points.T)
    # Get the eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # The eigenvector with the largest eigenvalue gives the main axis orientation
    main_axis = eigenvectors[:, np.argmax(eigenvalues)]
    # Calculate the angle of rotation
    angle = np.arctan2(main_axis[1], main_axis[0])
    return angle

def get_regular_polygon_symmetry(points, n_sides):
    center = np.mean(points, axis=0)
    angle = calculate_orientation(points)
    
    symmetry_lines = []
    for i in range(n_sides):
        line_angle = angle + i * np.pi / n_sides
        dx = np.cos(line_angle)
        dy = np.sin(line_angle)
        
        if abs(dx) < 1e-6:  # Nearly vertical line
            symmetry_lines.append((np.inf, center[0]))
        else:
            slope = dy / dx
            intercept = center[1] - slope * center[0]
            symmetry_lines.append((slope, intercept))
    
    return symmetry_lines

def fit_bezier_curve(points, degree=3):
    def bezier(t, control_points):
        n = len(control_points) - 1
        return sum(scipy.special.comb(n, i) * (1-t)**(n-i) * t**i * control_points[i] for i in range(n+1))

    def error_func(params):
        control_points = np.reshape(params, (degree+1, 2))
        return np.linalg.norm(bezier(t, control_points) - points, axis=1)

    t = np.linspace(0, 1, len(points))
    initial_control_points = np.linspace(points[0], points[-1], degree+1)
    result = least_squares(error_func, initial_control_points.ravel())
    control_points = np.reshape(result.x, (degree+1, 2))

    bezier_points = bezier(t, control_points)
    bezier_error = calculate_polygon_error(points, bezier_points)

    _, _, symmetries = fit_line(bezier_points)

    return bezier_error, bezier_points, symmetries

def fit_b_spline(points, degree=3):
    t = np.linspace(0, 1, len(points))
    tck = splrep(t, points.T, k=degree)
    spline_points = np.column_stack(splev(t, tck))
    spline_error = calculate_polygon_error(points, spline_points)
    _, _, symmetries = fit_line(spline_points)
    return spline_error, spline_points, symmetries

# Sample usage
# filename = 'problems/problems/isolated.csv'
# polylines = read_csv(filename)
# shapes, names, symmetries = fit_shapes(polylines)
# plot(shapes, filename.split('/')[-1][:-4] + '.png', names, symmetries)
