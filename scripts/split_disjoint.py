import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from helper import read_csv
from detect_shapes import fit_line

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

def split_polylines_to_disjoint(polylines):
    lines = [poly.tolist() for polyline in polylines for poly in polyline]

    while True:
        f = False
        for i in range(len(lines)):
            cur_line = lines[i]
            if len(cur_line) < 4:
                continue
            for j in range(len(cur_line) - 2):
                p1, p2, p3 = cur_line[j], cur_line[j+1], cur_line[j+2]
                angle_diff = angle_difference(calculate_angle(calculate_slope(p1, p2)), calculate_angle(calculate_slope(p2, p3)))
                if angle_diff >= 30:
                    new_line_1 = cur_line[:j+2]
                    new_line_2 = cur_line[j+2:]
                    if len(new_line_1) < 4:
                        e1 = 0.9
                    else:
                        e1, _, _ = fit_line(np.array(new_line_1))
                    if len(new_line_2) < 4:
                        e2 = 0.9
                    else:
                        e2, _, _ = fit_line(np.array(new_line_2))
                    if e1 < 5 and e2 < 5:
                        lines[i] = new_line_1
                        lines.insert(i+1, new_line_2)
                        f = True
                        break
            if f:
                break
        
        if not f:
            break
    
    disjoint_polylines = []
    # Convert polylines to shapely LineStrings
    lines = [LineString(np.array(polyline)) for polyline in lines if len(polyline) > 1]
    disjoint_polylines = []
    pairs_checked = set()

    while True:
        f = False
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                if (i, j) not in pairs_checked and lines[i].intersects(lines[j]) and not lines[i].touches(lines[j]):
                    pairs_checked.add((i, j))
                    intersection = lines[i].intersection(lines[j])
                    new_i = lines[i].difference(intersection)
                    new_j = lines[j].difference(intersection)
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
# path = r'problems\problems\frag2.csv'
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
#     midpoint = np.mean(polyline, axis=0)
#     plt.text(midpoint[0], midpoint[1], f'{i}', fontsize=12)
    
# plt.show()