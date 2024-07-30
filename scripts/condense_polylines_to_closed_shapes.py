from helper_for_csvs import read_csv, plot
import numpy as np
from shapely.geometry import LineString

def check_overlap(polyline1, polyline2):
    line1 = LineString([tuple(point) for point in polyline1[0]])
    line2 = LineString([tuple(point) for point in polyline2[0]])
    intersection = line1.intersection(line2)
    
    if intersection.is_empty:
        return False, None
    else:
        # intersection can be a Point, MultiPoint, LineString, or MultiLineString
        if intersection.geom_type == 'Point':
            return True, [intersection.coords[:]]
        elif intersection.geom_type == 'MultiPoint':
            return True, [point.coords[:] for point in intersection]
        elif intersection.geom_type == 'LineString' or intersection.geom_type == 'MultiLineString':
            return True, [list(intersection.coords)]
        else:
            return True, None  # Handle other types if necessary

def find(i, parent):
    if parent[i] == i:
        return i
    return find(parent[i], parent)

def union(i, j, parent, polylines, point):
    def is_in_first_half(polyline, point):
        midpoint = len(polyline) // 2
        first_half = polyline[:midpoint]
        return tuple(point) in [tuple(p) for p in first_half]

    in_first_half_i = is_in_first_half(polylines[i], point)
    in_first_half_j = is_in_first_half(polylines[j], point)

    if in_first_half_i and not in_first_half_j:
        parent[find(j, parent)] = find(i, parent)
    elif in_first_half_j and not in_first_half_i:
        parent[find(i, parent)] = find(j, parent)
    else:
        # If the point is in the first half of both or neither, choose i as parent by default
        parent[find(j, parent)] = find(i, parent)

def unique_points(points):
    seen = set()
    unique_list = []
    for point in points:
        tuple_point = tuple(point)  # Convert to a tuple to make it hashable
        if tuple_point not in seen:
            seen.add(tuple_point)
            unique_list.append(point)
    unique_list.sort(key=lambda p: (p[0], p[1]))
    return unique_list

def find_overlaps(polylines):
    parent = list(range(len(polylines)))
    for i, polyline1 in enumerate(polylines):
        for j, polyline2 in enumerate(polylines):
            if i != j:
                intersection = check_overlap(polyline1, polyline2)
                if intersection[0]:
                    union(i, j, parent, polylines, intersection[1])
    
    overlaps = {}
    for i in range(len(polylines)):
        parent_i = find(i, parent)
        if parent_i not in overlaps:
            overlaps[parent_i] = list()
        overlaps[parent_i].extend(polylines[i][0])

    result = list(overlaps.values())
    res = [[] for _ in range(len(result))]
    for i in range(len(result)):
        res[i].append(np.array(unique_points(result[i])))
    return res


csv_path = r'C:\Users\GURDARSH VIRK\OneDrive\Documents\adobe-gensolve-2024\problems\problems\frag0.csv'
paths = read_csv(csv_path)
closed_shapes = find_overlaps(paths)
# print(closed_shapes)
plot(closed_shapes)