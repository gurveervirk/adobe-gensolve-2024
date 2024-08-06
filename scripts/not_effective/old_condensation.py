from shapely.geometry import LineString, Polygon, MultiPolygon, Point
from shapely.ops import polygonize, unary_union
# from shapely import GeometryCollection, node, intersection
from helper_for_csvs import read_csv
from matplotlib import pyplot as plt

def find_overlaps(paths):
    # Flatten the list of polylines
    all_lines = [LineString(polyline) for polyline_group in paths for polyline in polyline_group]
    
    # Find all intersection points
    union_lines = unary_union(all_lines)
    # Find closed shapes
    closed_shapes = list(polygonize(union_lines))
    
    return closed_shapes

def plot(shapes):
    fig, ax = plt.subplots()
    for shape in shapes:
        if isinstance(shape, Polygon) or isinstance(shape, MultiPolygon):
            if isinstance(shape, MultiPolygon):
                for poly in shape:
                    x, y = poly.exterior.xy
                    ax.plot(x, y)
            else:
                x, y = shape.exterior.xy
                ax.plot(x, y)
        elif isinstance(shape, LineString):
            x, y = shape.xy
            ax.plot(x, y)
    plt.show()

csv_path = r'path/to/csv'
paths = read_csv(csv_path)
print("Number of original paths:", len(paths))
closed_shapes = find_overlaps(paths)
print("Number of merged shapes:", len(closed_shapes))
# plot(closed_shapes)



# use following to find gaps in the lines
    # lines =  [line for line in all_lines]
    # for i, line in enumerate(lines):  
    #     # go through each line added first to second
    #     # then second to third and so on
    #     shply_lines = lines[:i] + lines[i+1:]
    #     # 0 is start point and -1 is end point
    #     # run through
    #     for start_end in [0, -1]:
    #         # convert line to point
    #         pt = Point(line.coords[start_end])
    #         # replace touch in the original script to avoid floating point problems
    #         if any(pt.distance(next_line) == 0.0 for next_line in shply_lines):
    #             continue
    #         else:
    #             print(pt)