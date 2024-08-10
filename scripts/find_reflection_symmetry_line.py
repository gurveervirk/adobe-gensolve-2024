import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from shapely.geometry import Polygon, Point
from helper import read_csv

# Function to calculate the centroid of the points
def find_centroid(points):
    # points should be a 2D array of shape (n, 2) where n is the number of points
    centroid = np.mean(points, axis=0)
    return centroid

def calculate_polygon_error(original_points, fitted_points):
    fitted_polygon = Polygon(fitted_points).exterior
    distances = [fitted_polygon.distance(Point(p)) for p in original_points]
    return np.mean(distances)

# Function to reflect points around a line passing through the centroid at a given angle
def reflect_points(points, centroid, angle):
    angle_rad = np.deg2rad(angle)
    
    # Translate points to centroid
    translated_points = points - centroid
    
    # Rotation matrix to align the reflection axis with the x-axis
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad),  np.cos(angle_rad)]])
    
    # Rotate points so that the reflection line aligns with the x-axis
    rotated_points = translated_points @ rotation_matrix.T
    
    # Reflect across the x-axis
    reflected_rotated_points = rotated_points.copy()
    reflected_rotated_points[:, 1] = -reflected_rotated_points[:, 1]
    
    # Rotate back to the original orientation
    inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
    reflected_points = reflected_rotated_points @ inverse_rotation_matrix.T
    
    # Translate back to the original centroid
    reflected_points += centroid
    
    return reflected_points

# Helper function to calculate symmetry score for a given angle
def symmetry_score_for_angle(angle, points, centroid):
    reflected_points = reflect_points(points, centroid, angle)
    return calculate_polygon_error(points, reflected_points), angle

# Function to find the best angle that represents the line of symmetry using parallel processing
def find_reflection_symmetry_parallel(points, centroid):
    angles = np.linspace(0, 180, num=1000)
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(symmetry_score_for_angle, [(angle, points, centroid) for angle in angles])
    
    best_symmetry_score, best_angle = min(results, key=lambda x: x[0])
    
    # Calculate the slope and intercept of the line of symmetry
    best_angle_rad = np.deg2rad(best_angle)
    slope = np.tan(best_angle_rad)
    intercept = centroid[1] - slope * centroid[0]
    
    print(f"Best symmetry score: {best_symmetry_score}")
    return (slope, intercept), best_symmetry_score

# Function to plot the points and the line of symmetry
def plot_symmetry_line(points, centroid, slope, intercept):
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], label='Original Points')
    plt.scatter(centroid[0], centroid[1], color='red', label='Centroid')
    
    # Calculate the endpoints of the symmetry line
    x_values = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), num=100)
    y_values = slope * x_values + intercept
    
    plt.plot(x_values, y_values, 'r--', label=f'Symmetry Line (slope={slope:.3f}, intercept={intercept:.3f})')
    
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Example usage
# if __name__ == '__main__':
#     points = read_csv(r'problems\problems\occlusion1.csv')[1][1]

#     centroid = find_centroid(points)
#     slope, intercept = find_reflection_symmetry_parallel(points, centroid)

#     print(f"The line of symmetry has a slope of {slope:.3f} and an intercept of {intercept:.3f}")

#     # Plot the points and the symmetry line
#     plot_symmetry_line(points, centroid, slope, intercept)
