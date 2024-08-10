from split_disjoint import split_polylines_to_disjoint, extend_and_connect_polylines
from helper import read_csv, plot_simple, plot
from matplotlib import pyplot as plt
from detect_shapes import fit_shape, calculate_polygon_error, fit_line
import numpy as np

# Try fitting shapes to the connected polylines for completion
def complete_curves(polylines):
    result = []
    i = 0
    for polyline in polylines:
        if result:
            flag = False
            for fitted_shapes in result:
                if fitted_shapes[3] == 'line':
                    m, b = fitted_shapes[4][0]
                    mean_distance = np.mean(np.abs(polyline[:, 1] - m * polyline[:, 0] - b))
                    if mean_distance < 5:
                        new_points = np.vstack((fitted_shapes[1], polyline))
                        _, best_points, symmetry_lines = fit_line(new_points)
                        result[fitted_shapes[0]] = [fitted_shapes[0], new_points, best_points, "line", symmetry_lines]
                        flag = True
                        break

                if calculate_polygon_error(polyline, fitted_shapes[2]) < 10:
                    new_points = np.vstack((fitted_shapes[1], polyline))
                    best_points, _, best_shape, symmetry_lines = fit_shape(new_points)
                    result[fitted_shapes[0]] = [fitted_shapes[0], new_points, best_points, best_shape, symmetry_lines]
                    flag = True
                    break
            if flag:
                continue
        
        best_points, lowest_error, best_shape, symmetry_lines = fit_shape(polyline)
        if lowest_error < 10:
            result.append([i, polyline, best_points, best_shape, symmetry_lines])
            i += 1
            
    return result

if __name__ == '__main__':
    # Load the data
    path = r'problems\problems\frag2.csv'
    polylines = read_csv(path)

    # Split into disjoint polylines
    disjoint_polylines = split_polylines_to_disjoint(polylines)

    # Connect disjoint polylines naturally
    connected_polylines = extend_and_connect_polylines(disjoint_polylines)

    result = complete_curves(connected_polylines)
    completed_polylines = [r[2] for r in result]
    names = [r[3] for r in result]
    best_symmetry_lines = [r[4] for r in result]

    plot_simple([completed_polylines], path.split('\\')[-1].split('.')[0] + '_completed.png')
    # plot([completed_polylines], path.split('\\')[-1].split('.')[0] + '_completed.png', names, best_symmetry_lines)