from split_disjoint import split_polylines_to_disjoint, extend_and_connect_polylines
from helper import read_csv, plot_simple
from matplotlib import pyplot as plt
from detect_shapes import fit_shape, calculate_polygon_error

# Load the data
path = r'problems\problems\occlusion1.csv'
polylines = read_csv(path)

# Split into disjoint polylines
disjoint_polylines = split_polylines_to_disjoint(polylines)

# Connect disjoint polylines naturally
connected_polylines = extend_and_connect_polylines(disjoint_polylines)

# Try fitting shapes to the connected polylines for completion
def complete_curves(polylines):
    result = []
    for polyline in polylines:
        if result:
            flag = False
            for fitted_shapes in result:
                if calculate_polygon_error(polyline, fitted_shapes) < 10:
                    flag = True
                    break
            if flag:
                continue
        
        best_points, lowest_error, best_shape, symmetry_lines = fit_shape(polyline)
        if lowest_error < 10:
            result.append(best_points)
            
    return result

completed_polylines = complete_curves(connected_polylines)
plot_simple([completed_polylines], path.split('\\')[-1].split('.')[0] + '_completed.png')