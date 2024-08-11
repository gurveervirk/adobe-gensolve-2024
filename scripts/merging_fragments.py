
import numpy as np
from shapely.geometry import Polygon, LineString
from helper_for_csvs import read_csv, plot, plot_simple
from detect_shapes import fit_shape, fit_line
import numpy as np
from shapely.geometry import LineString, Polygon, Point, MultiPoint, MultiLineString
from shapely.ops import unary_union, polygonize
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial import distance
import cv2
import traceback
from scipy.optimize import least_squares
threshold = 10


import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

def is_closed_shape_with_convex_hull(points):
    # Convert points to a NumPy array if they are not already
    points = np.array(points)
    
    # Compute the convex hull
    hull = ConvexHull(points)
    
    # Extract the vertices of the convex hull
    hull_points = points[hull.vertices]
    
    # Create a Polygon from the hull points
    hull_polygon = Polygon(hull_points)
    
    # Check if the polygon is valid and has a non-zero area
    if hull_polygon.is_valid and hull_polygon.area > 0:
        return True  # Return the points that form the closed shape
    else:
        return False
def merge_polylines_combinations(polylines):
    
    line_strings = [LineString(polyline) for polyline in polylines]
    used_indices = set()
    shapes_detected = []
    shape_points = []
    remaining_polylines = []

    for r in range(1,8):
        if r==2:
            continue
        combinations = list(itertools.combinations(enumerate(line_strings), r))
        for combo in combinations:
            indices, combo_lines = zip(*combo)
            if any(index in used_indices for index in indices):
                continue
            # try:
            merged = unary_union(combo_lines)
            points=[]
            if(isinstance(merged, MultiLineString)):
                # combined_coords = []
                # for line in merged.geoms:
                #     combined_coords.extend(line.coords)
                
                # # Create a new LineString from the combined coordinates
                # merged_line = LineString(combined_coords)
                # if(not merged.is_ring or not merged.is_closed):
                #     continue
                closed_lines = [line for line in merged.geoms]
                for line in closed_lines:
                    points.extend(list(line.coords))
                if(not is_closed_shape_with_convex_hull(points)):
                    continue
                best_points, lowest_error, best_shape, symmetry_lines = fit_shape(points)
                if lowest_error<5:
                    used_indices.update(indices)
                    shapes_detected.append(best_shape)
                    shape_points.append(best_points)
                
            if(isinstance(merged, LineString)):
                    points =np.array([list(coord) for coord in merged.coords])
                    lowest_error, best_points, symmetry_lines = fit_line(points)
                    if lowest_error<10:
                        used_indices.update(indices)
                        shapes_detected.append("line")
                        shape_points.append(best_points)
        for i in range(len(line_strings)):
            if i not in used_indices:
                print(len(remaining_polylines))
                remaining_polylines.append(np.array([list(coord) for coord in line_strings[i].coords]))
        # except Exception as e:
            #     print(f"Error merging combination {combo}: {e}")

    # print(shapes_detected)
    return shapes_detected,shape_points, remaining_polylines
# Main script
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_shapes(detected_shapes, shape_names, remaining_polylines, original_polylines, output_path):
    plt.figure()
    # Plot detected shapes with their corresponding names and colors
    i=0
    for shape, name in zip(detected_shapes, shape_names):
        shape = np.array(shape)
        plt.plot(shape[:, 0], shape[:, 1], label=name)
        
    
    # Plot original polylines with the same color as their detected shape
    # for shape, name in zip(detected_shapes, shape_names):
    #     for polyline in original_polylines:
    #         # if np.array_equal(np.array(polyline), np.array(shape)):
    #             polyline = np.array(polyline)
    #             plt.plot(polyline[:, 0], polyline[:, 1],  linestyle='--')
    
    # # Plot remaining polylines with a different style
    # remaining_polylines = np.array(remaining_polylines)
    print(len(remaining_polylines))
    for polyline in remaining_polylines:
        polyline = np.array(polyline)
        plt.plot(polyline[:, 0], polyline[:, 1], linestyle='--' ,label='Remaining')
    # plt.legend()
    plt.savefig(output_path)
    plt.show()

# Main script
filename = 'problems/problems/frag1.csv'
polylines = read_csv(filename)
# Flatten the list of polylines for intersection checking
flattened_polylines = [item for sublist in polylines for item in sublist]

# Check if merged polylines form shapes
names,shapes, remaining_polylines = merge_polylines_combinations(flattened_polylines)
# print(type(shapes))
# Plot shapes and remaining polylines
plot_shapes(shapes, names, remaining_polylines, flattened_polylines, filename.replace('.csv', '.png'))

# plot([shapes], filename.split('/')[-1][:-4] + '_shapes.png', names,symmetries)
# plot_simple([remaining_polylines], filename.split('/')[-1][:-4] + '_remaining.png')
