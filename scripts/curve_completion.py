from split_disjoint import split_polylines_to_disjoint, extend_and_connect_polylines, points_are_close
from helper import read_csv, plot_simple, plot
from matplotlib import pyplot as plt
from detect_shapes import fit_shape, calculate_polygon_error, fit_line
import numpy as np
import math

# Try fitting shapes to the connected polylines for completion
def complete_curves(polylines):
    result = []
    polylines.sort(key=lambda x: len(x), reverse=True)

    for i, polyline in enumerate(polylines):
        if result:
            flag = False
            for j, fitted_shapes in enumerate(result):
                if fitted_shapes[3] == 'line':
                    m, b = fitted_shapes[4][0]
                    mean_distance = np.mean(np.abs(polyline[:, 1] - m * polyline[:, 0] - b))
                    if mean_distance < 10:
                        e11, e12 = polyline[0], polyline[-1]
                        e21, e22 = fitted_shapes[1][0], fitted_shapes[1][-1]

                        dist_e11_e21 = np.linalg.norm(e11 - e21)
                        dist_e11_e22 = np.linalg.norm(e11 - e22)
                        dist_e12_e21 = np.linalg.norm(e12 - e21)
                        dist_e12_e22 = np.linalg.norm(e12 - e22)

                        distances = [dist_e11_e21, dist_e11_e22, dist_e12_e21, dist_e12_e22]
                        min_distance_index = np.argmin(distances)

                        if min_distance_index == 0:
                            new_points = np.concatenate((polyline[::-1], fitted_shapes[1]), axis=0)
                        elif min_distance_index == 1:
                            new_points = np.concatenate((fitted_shapes[1], polyline), axis=0)
                        elif min_distance_index == 2:
                            new_points = np.concatenate((polyline, fitted_shapes[1]), axis=0)
                        else:
                            new_points = np.concatenate((polyline, fitted_shapes[1][::-1]), axis=0)
                        
                        if len(new_points) > 2:
                            # Check if the polyline is truly a valid line
                            error, best_points, cur_slope_intercept = fit_line(polyline)
                            if error < 5:
                                fit_slope_intercept = fitted_shapes[4][0]
                                fit_slope = fit_slope_intercept[0]
                                cur_slope = cur_slope_intercept[0][0]

                                # Calculate the angle between the two slopes
                                angle_radians = math.atan(abs((fit_slope - cur_slope) / (1 + fit_slope * cur_slope)))

                                # Convert angle from radians to degrees
                                angle_degrees = math.degrees(angle_radians)

                                # Check if the angle is greater than or equal to 30 degrees
                                if angle_degrees >= 40:
                                    # Continue with your process
                                    continue

                        _, best_points, symmetry_lines = fit_line(new_points)
                        fitted_shapes[0] = fitted_shapes[0].union({i})
                        result[j] = [fitted_shapes[0], new_points, best_points, "line", symmetry_lines]
                        flag = True
                        break

                elif calculate_polygon_error(polyline, fitted_shapes[2]) < 10:
                    e11, e12 = polyline[0], polyline[-1]
                    e21, e22 = fitted_shapes[1][0], fitted_shapes[1][-1]
                    if points_are_close(e11, e21):
                        new_points = np.concatenate((polyline, fitted_shapes[1][::-1]), axis=0)
                    elif points_are_close(e11, e22):
                        new_points = np.concatenate((polyline, fitted_shapes[1]), axis=0)
                    elif points_are_close(e12, e21):
                        new_points = np.concatenate((fitted_shapes[1], polyline), axis=0)
                    elif points_are_close(e12, e22):
                        new_points = np.concatenate((fitted_shapes[1][::-1], polyline), axis=0)
                    else:
                        new_points = np.concatenate((fitted_shapes[1], polyline), axis=0)
                    best_points, _, best_shape, symmetry_lines = fit_shape(new_points)
                    fitted_shapes[0] = fitted_shapes[0].union({i})
                    result[j] = [fitted_shapes[0], new_points, best_points, best_shape, symmetry_lines]
                    flag = True
                    break
            if flag:
                continue
        
        best_points, lowest_error, best_shape, symmetry_lines = fit_shape(polyline)
        if lowest_error < 10:
            result.append([{i}, polyline, best_points, best_shape, symmetry_lines])

    return result

if __name__ == '__main__':
    # Load the data
    path = r'problems\problems\occlusion2.csv'
    polylines = read_csv(path)

    # Split into disjoint polylines
    disjoint_polylines = split_polylines_to_disjoint(polylines)

    # Connect disjoint polylines naturally
    connected_polylines = extend_and_connect_polylines(disjoint_polylines)

    result = complete_curves(connected_polylines)
    for i, r in enumerate(result):
        print(f"Completed curve {i}: {r[0]}")

    completed_polylines = [r[2] for r in result]
    names = [r[3] for r in result]
    best_symmetry_lines = [r[4] for r in result]

    # Preferably use plot_simple for simple plots
    # plot_simple([completed_polylines], path.split('\\')[-1].split('.')[0] + '_completed.png')

    # Preferably use plot for complex plots
    plot([completed_polylines], path.split('\\')[-1].split('.')[0] + '_completed.png', names, best_symmetry_lines)