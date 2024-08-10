import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from helper import read_csv

def split_polylines_to_disjoint(polylines):
    disjoint_polylines = []
    
    # Convert polylines to shapely LineStrings
    lines = [LineString(poly) for polyline in polylines for poly in polyline]
    disjoint_polylines = []

    while True:
        f = False
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                if lines[i].intersects(lines[j]) and not lines[i].touches(lines[j]):
                    intersection = lines[i].intersection(lines[j])
                    new_i = lines[i].difference(lines[j])
                    new_j = lines[j].difference(lines[i])
                    lines[i] = new_i
                    lines[j] = new_j
                    lines.append(intersection)
                    f = True
                    print(f"Splitting lines {i} and {j}")
                    break
            if f:
                break
        
        if not f:
            break
    
    for line in lines:
        if line.geom_type == 'MultiLineString':
            for l in line.geoms:
                disjoint_polylines.append(np.array(l.coords))
        elif line.geom_type == 'LineString':
            disjoint_polylines.append(np.array(line.coords))

    return disjoint_polylines

def points_are_close(p1, p2, tol=1e-5):
    distance = np.linalg.norm(np.array(p1) - np.array(p2))
    return distance < tol

def calculate_slope(p1, p2):
    """ Calculate the slope of the line segment between points p1 and p2 """
    if p2[0] == p1[0]:  # Handle vertical lines
        return np.inf
    return (p2[1] - p1[1]) / (p2[0] - p1[0])

def calculate_angle(slope):
    """ Calculate the angle of the slope with respect to the x-axis """
    return np.arctan(slope) * (180 / np.pi)  # Convert radians to degrees

def angle_difference(angle1, angle2):
    """ Calculate the smallest difference between two angles """
    diff = abs(angle1 - angle2)
    return min(diff, 360 - diff)  # Adjust for angle wrap-around

def count_close_points(polylines, point, tolerance=1e-5):
    count = -1
    for polyline in polylines:
        if points_are_close(point, polyline[0], tolerance) or points_are_close(point, polyline[-1], tolerance):
            count += 1

    print(f"Close points to {point}: {count}")
    return count

def merge_close_points_if_unique(extended_polylines, tolerance=1e-5):
    while True:
        merged_polylines = []
        visited = set()
        merge_occurred = False

        for i, polyline_i in enumerate(extended_polylines):
            if i in visited:
                continue
            
            merged = False
            for j, polyline_j in enumerate(extended_polylines):
                if i != j and j not in visited:
                    start_i, end_i = polyline_i[0], polyline_i[-1]
                    start_j, end_j = polyline_j[0], polyline_j[-1]

                    # Check if start of polyline_i is close to polyline_j
                    if points_are_close(start_i, start_j, tolerance) and count_close_points(extended_polylines, start_i, tolerance) == 1:
                        new_polyline = np.vstack([polyline_j[::-1], polyline_i[1:]])
                        merged_polylines.append(new_polyline)
                        visited.add(i)
                        visited.add(j)
                        merged = True
                        merge_occurred = True
                        break
                    elif points_are_close(start_i, end_j, tolerance) and count_close_points(extended_polylines, start_i, tolerance) == 1:
                        new_polyline = np.vstack([polyline_j, polyline_i[1:]])
                        merged_polylines.append(new_polyline)
                        visited.add(i)
                        visited.add(j)
                        merged = True
                        merge_occurred = True
                        break
                    elif points_are_close(end_i, start_j, tolerance) and count_close_points(extended_polylines, end_i, tolerance) == 1:
                        new_polyline = np.vstack([polyline_i, polyline_j[1:]])
                        merged_polylines.append(new_polyline)
                        visited.add(i)
                        visited.add(j)
                        merged = True
                        merge_occurred = True
                        break
                    elif points_are_close(end_i, end_j, tolerance) and count_close_points(extended_polylines, end_i, tolerance) == 1:
                        new_polyline = np.vstack([polyline_i, polyline_j[::-1][1:]])
                        merged_polylines.append(new_polyline)
                        visited.add(i)
                        visited.add(j)
                        merged = True
                        merge_occurred = True
                        break

            if not merged:
                merged_polylines.append(polyline_i)

        if not merge_occurred:
            break

        extended_polylines = merged_polylines

    return merged_polylines

def extend_and_connect_polylines(disjoint_polylines, angle_threshold=30):
    visited = set()
    extended_polylines = []

    for idx, polyline in enumerate(disjoint_polylines):
        if len(polyline) > 0 and idx not in visited:
            visited.add(idx)
            
            # Start with the two edge points of the selected polyline
            start_point, end_point = polyline[0], polyline[-1]
            start_slope = calculate_slope(polyline[0], polyline[1])
            end_slope = calculate_slope(polyline[-2], polyline[-1])
            
            current_polyline = polyline
            
            while True:
                extended = False
                for other_idx, other_polyline in enumerate(disjoint_polylines):
                    if other_idx not in visited and len(other_polyline) > 0:
                        other_start, other_end = other_polyline[0], other_polyline[-1]
                        
                        # Check if the other polyline shares a point with the selected one
                        if np.array_equal(other_start, start_point) or points_are_close(other_start, start_point):
                            # Compare slopes from start_point
                            other_slope = calculate_slope(other_start, other_polyline[1])
                            angle1 = calculate_angle(start_slope)
                            angle2 = calculate_angle(other_slope)
                            angle_diff = angle_difference(angle1, angle2)
                            
                            if angle_diff < angle_threshold:
                                visited.add(other_idx)
                                current_polyline = np.vstack([current_polyline[::-1], other_polyline])
                                start_point = current_polyline[0]
                                end_point = current_polyline[-1]
                                start_slope = calculate_slope(current_polyline[0], current_polyline[1])
                                end_slope = calculate_slope(current_polyline[-2], current_polyline[-1])
                                extended = True
                                break
                            
                        elif np.array_equal(other_end, start_point) or points_are_close(other_end, start_point):
                            # Compare slopes from start_point
                            other_slope = calculate_slope(other_polyline[-2], other_end)
                            angle1 = calculate_angle(start_slope)
                            angle2 = calculate_angle(other_slope)
                            angle_diff = angle_difference(angle1, angle2)
                            
                            if angle_diff < angle_threshold:
                                visited.add(other_idx)
                                current_polyline = np.vstack([current_polyline[::-1], other_polyline[::-1]])
                                start_point = current_polyline[0]
                                end_point = current_polyline[-1]
                                start_slope = calculate_slope(current_polyline[0], current_polyline[1])
                                end_slope = calculate_slope(current_polyline[-2], current_polyline[-1])
                                extended = True
                                break

                        elif np.array_equal(other_start, end_point) or points_are_close(other_start, end_point):
                            # Compare slopes from end_point
                            other_slope = calculate_slope(other_start, other_polyline[1])
                            angle1 = calculate_angle(end_slope)
                            angle2 = calculate_angle(other_slope)
                            angle_diff = angle_difference(angle1, angle2)
                            
                            if angle_diff < angle_threshold:
                                visited.add(other_idx)
                                current_polyline = np.vstack([current_polyline, other_polyline])
                                start_point = current_polyline[0]
                                end_point = current_polyline[-1]
                                start_slope = calculate_slope(current_polyline[0], current_polyline[1])
                                end_slope = calculate_slope(current_polyline[-2], current_polyline[-1])
                                extended = True
                                break
                            
                        elif np.array_equal(other_end, end_point) or points_are_close(other_end, end_point):
                            # Compare slopes from end_point
                            other_slope = calculate_slope(other_polyline[-2], other_end)
                            angle1 = calculate_angle(end_slope)
                            angle2 = calculate_angle(other_slope)
                            angle_diff = angle_difference(angle1, angle2)
                            
                            if angle_diff < angle_threshold:
                                visited.add(other_idx)
                                current_polyline = np.vstack([current_polyline, other_polyline[::-1]])
                                start_point = current_polyline[0]
                                end_point = current_polyline[-1]
                                start_slope = calculate_slope(current_polyline[0], current_polyline[1])
                                end_slope = calculate_slope(current_polyline[-2], current_polyline[-1])
                                extended = True
                                break

                if not extended:
                    break
            
            extended_polylines.append(current_polyline)
    
    # Add polylines that were not extended
    for idx, polyline in enumerate(disjoint_polylines):
        if idx not in visited and len(polyline) > 0:
            extended_polylines.append(polyline)
    
    # Remove empty polylines
    extended_polylines = [polyline for polyline in extended_polylines if len(polyline) > 0]

    # Perform the final merge of close points only if they are unique
    final_polylines = merge_close_points_if_unique(extended_polylines)

    return final_polylines

# Sample usage
# path = r'problems\problems\occlusion2.csv'
# polylines = read_csv(path)
# disjoint_polylines = split_polylines_to_disjoint(polylines)
# extended_polylines = extend_and_connect_polylines(disjoint_polylines)

# print(f"Original polylines: {len(polylines)}")
# print(f"Disjoint polylines: {len(disjoint_polylines)}")
# print(f"Extended polylines: {len(extended_polylines)}")

# # Visualization
# plt.figure()

# for i, polyline in enumerate(extended_polylines):
#     plt.plot(polyline[:, 0], polyline[:, 1], label=f'Polyline {i}')
#     midpoint = polyline[len(polyline)//2]
#     plt.text(midpoint[0], midpoint[1], f'{i}', fontsize=12)
    
# plt.show()