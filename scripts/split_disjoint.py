import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from helper_for_csvs import read_csv

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
    
    print(f"Number of disjoint polylines: {len(lines)}")
    for line in lines:
        if line.geom_type == 'MultiLineString':
            for l in line.geoms:
                disjoint_polylines.append(np.array(l.coords))
        elif line.geom_type == 'LineString':
            disjoint_polylines.append(np.array(line.coords))

    return disjoint_polylines



path = r'problems\problems\occlusion2.csv'
polylines = read_csv(path)
disjoint_polylines = split_polylines_to_disjoint(polylines)

# Plotting the result
plt.figure()
for polyline in disjoint_polylines:
    if len(polyline) > 0:
        plt.plot(polyline[:, 0], polyline[:, 1])
plt.show()
