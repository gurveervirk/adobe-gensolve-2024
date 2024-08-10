from split_disjoint import split_polylines_to_disjoint, extend_and_connect_polylines
from helper import read_csv, plot_simple, plot
from matplotlib import pyplot as plt
from detect_shapes import fit_shape, calculate_polygon_error

# Try fitting shapes to the connected polylines for completion
def complete_curves(polylines):
    result = []
    names = []
    best_symmetry_lines = []
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
            names.append(best_shape)
            best_symmetry_lines.append(symmetry_lines)
            
    return result, names, best_symmetry_lines

if __name__ == '__main__':
    # Load the data
    path = r'problems\problems\occlusion2.csv'
    polylines = read_csv(path)

    # Split into disjoint polylines
    disjoint_polylines = split_polylines_to_disjoint(polylines)

    # Connect disjoint polylines naturally
    connected_polylines = extend_and_connect_polylines(disjoint_polylines)

    completed_polylines, names, best_symmetry_lines = complete_curves(connected_polylines)
    
    # plot_simple([completed_polylines], path.split('\\')[-1].split('.')[0] + '_completed.png')
    plot([completed_polylines], path.split('\\')[-1].split('.')[0] + '_completed.png', names, best_symmetry_lines)