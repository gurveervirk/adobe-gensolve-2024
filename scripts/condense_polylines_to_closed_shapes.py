from helper_for_csvs import read_csv, plot
from shapely.geometry import LineString, Point, Polygon
from matplotlib import pyplot as plt
import numpy as np
from shapely.ops import nearest_points
from detect_shapes import fit_shape

def find_intersections_and_update_paths(paths):
    lines = [LineString(path[0]) for path in paths]
    
    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines):
            if i >= j:
                continue
            
            inters = line1.intersection(line2)
            if isinstance(inters, Point):
                print(i, j, inters)
                # Add intersection point to both lines
                if not Point(inters).within(line1):
                    print('here i')
                    path1_coords = list(line1.coords)
                    path1_coords.append((inters.x, inters.y))
                    paths[i][0] = np.array(sorted(path1_coords, key=lambda coord: line1.project(Point(coord))))
                    lines[i] = LineString(paths[i][0])
                
                if not Point(inters).within(line2):
                    print('here j')
                    path2_coords = list(line2.coords)
                    path2_coords.append((inters.x, inters.y))
                    paths[j][0] = np.array(sorted(path2_coords, key=lambda coord: line2.project(Point(coord))))
                    lines[j] = LineString(paths[j][0])

            elif inters.geom_type == 'MultiPoint':
                for inter in inters:
                    if not Point(inter).within(line1):
                        path1_coords = list(line1.coords)
                        path1_coords.append((inter.x, inter.y))
                        paths[i][0] = np.array(sorted(path1_coords, key=lambda coord: line1.project(Point(coord))))
                        lines[i] = LineString(paths[i][0])
                    
                    if not Point(inter).within(line2):
                        path2_coords = list(line2.coords)
                        path2_coords.append((inter.x, inter.y))
                        paths[j][0] = np.array(sorted(path2_coords, key=lambda coord: line2.project(Point(coord))))
                        lines[j] = LineString(paths[j][0])

    return lines

# def find_closest_points(lines, threshold=1e-2):
#     for i, line1 in enumerate(lines):
#         for j, line2 in enumerate(lines):
#             if i >= j:
#                 continue
            
#             # Find closest points between lines
#             nearest = nearest_points(line1, line2)
#             pt1, pt2 = nearest
            
#             distance = pt1.distance(pt2)
#             if distance < threshold and pt1 not in line2.coords:
#                 # Insert pt1 into line2 in sorted order
#                 coords = list(line2.coords)
#                 coords.append((pt1.x, pt1.y))
#                 coords = sorted(coords, key=lambda coord: line2.project(Point(coord)))
#                 lines[j] = LineString(coords)  # Update line2 with new point
    
#     return lines

def merge_polylines_to_closed_shapes(lines, fit_shape, error_threshold=500, threshold=1e-2):
    cur_best_error = float('inf')
    cur_best_points = None
    cur_best_indices = []
    def dfs(ci, cur_vis, visited, current_coords):
        nonlocal cur_best_error, cur_best_points, cur_best_indices
        current_line = lines[ci]
        cur_vis.append(ci)
        coords = list(current_coords)

        if ci != cur_vis[0] and len(cur_vis) > 2:
            starting_line = lines[cur_vis[0]]
            
            if current_line.intersects(starting_line):
                print(cur_vis)
                overlapping_point = current_line.intersection(starting_line)
                new_coords = coords.copy()
                index = new_coords.index(overlapping_point)
                new_coords = new_coords[index:]
                index = new_coords.index(overlapping_point[1:])
                new_coords = new_coords[:index+1]
                # print(len(new_coords))
                # closed_shape = Polygon(new_coords)
                # if closed_shape.is_valid:
                formatted_shape = np.array(new_coords)
                best_points, lowest_error = fit_shape(formatted_shape)
                if lowest_error < error_threshold and lowest_error < cur_best_error:
                    cur_best_error = lowest_error
                    cur_best_points = best_points
                    cur_best_indices = cur_vis.copy()
                    # return True
            
            else:
                nearest = nearest_points(current_line, starting_line)
                pt1, pt2 = nearest
                distance = pt1.distance(pt2)
                
                if distance < threshold:
                    new_coords = coords.copy()
                    new_coords.append(new_coords[0])
                    # closed_shape = Polygon(new_coords)
                    # if closed_shape.is_valid:
                    print(distance, cur_vis)
                    formatted_shape = np.array(new_coords)
                    best_points, lowest_error = fit_shape(formatted_shape)
                    if lowest_error < error_threshold and lowest_error < cur_best_error:
                        cur_best_error = lowest_error
                        cur_best_points = best_points
                        cur_best_indices = cur_vis.copy()
                    # return True
            
        for i in range(len(lines)):
            next_line = lines[i]
            if i in visited or i in cur_vis:
                continue

            if current_line.overlaps(next_line):
                overlapping_point = current_line.intersection(next_line)
                next_line_coords = list(next_line.coords)
                new_coords = coords.copy()
                index = new_coords.index(overlapping_point)
                if overlapping_point in new_coords[:len(new_coords)//2]:
                    new_coords = new_coords[index:]
                else:
                    new_coords = new_coords[:index+1]
                if overlapping_point in next_line_coords[:len(next_line_coords)//2]:
                    new_coords.extend(next_line_coords[next_line_coords.index(overlapping_point) + 1:])
                else:
                    new_coords.extend(next_line_coords[:next_line_coords.index(overlapping_point)])
                res = dfs(i, cur_vis, visited, new_coords)
                # if res:
                #     return True
                
            else:
                nearest = nearest_points(current_line, next_line)
                pt1, pt2 = nearest
                distance = pt1.distance(pt2)
                
                if distance < threshold:
                    next_line_coords = list(next_line.coords)
                    new_coords = coords.copy()
                    vis = cur_vis.copy()
                    if pt1 in new_coords[:len(new_coords)//2]:
                        new_coords = new_coords[::-1]
                        vis = vis[::-1]
                    if pt2 in next_line_coords[:len(next_line_coords)//2]:
                        new_coords.extend(next_line_coords)
                    else:
                        new_coords.extend(next_line_coords[::-1])
                    res = dfs(i, vis, visited, new_coords)
                    # if res:
                    #     return True
        
        cur_vis.remove(ci)
        # return False

    merged_shapes = []
    visited = set()
    
    for i in range(len(lines)):
        if lines[i] in visited:
            continue
        vis = []
        dfs(i, vis, visited, lines[i].coords)
        visited = set(list(visited) + cur_best_indices)
        if cur_best_error != float('inf'):
            merged_shapes.append(cur_best_points)
            cur_best_error = float('inf')
            cur_best_points = None
            cur_best_indices = []

    print(visited)
    return merged_shapes

def plot_paths(paths, intersections, close_points):
    plt.figure()
    
    # Plot each path with numbering
    for idx, path in enumerate(paths):
        plt.plot(path[:, 0], path[:, 1], marker='o', label=f'Path {idx + 1}')
        # Add a label next to the path
        mid_point = np.mean(path, axis=0)
        plt.text(mid_point[0], mid_point[1], f'Path {idx + 1}', fontsize=12, ha='right')
    
    # Plot intersection points
    for inter in intersections:
        plt.plot(inter.x, inter.y, 'ro')  # Plot intersection points in red
    
    # Plot close points
    for pt1, pt2 in close_points:
        plt.plot([pt1.x, pt2.x], [pt1.y, pt2.y], 'g--')  # Line between close points
        plt.plot(pt1.x, pt1.y, 'bo')  # Close point 1 in blue
        plt.plot(pt2.x, pt2.y, 'bs')  # Close point 2 in blue
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Paths, Intersection Points, and Close Points')
    plt.legend()
    plt.grid(True)
    plt.show()

csv_path = r'problems\problems\frag0.csv'
paths = read_csv(csv_path)
lines = find_intersections_and_update_paths(paths)
# lines = [LineString(path[0]) for path in paths]
# lines = find_closest_points(lines, threshold=0.1)
merged_shapes = merge_polylines_to_closed_shapes(lines, fit_shape, error_threshold=500, threshold=0.1)
print(len(merged_shapes))
plot([merged_shapes])