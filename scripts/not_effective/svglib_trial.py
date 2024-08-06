import numpy as np
import pandas as pd
from svgpathtools import svg2paths

def path_to_polylines(path, num_points=50):
    """Converts a Path object to a list of polylines with specified number of points."""
    total_length = path.length()
    distances = np.linspace(0, total_length, num_points)
    points = [path.point(path.ilength(d)) for d in distances]
    return points

def save_polylines_to_csv(polylines, output_file):
    """Saves the polylines to a CSV file in the format: polyline_index, 0, x, y."""
    data = []
    for i, polyline in enumerate(polylines):
        for point in polyline:
            data.append([i, 0, point.real, point.imag])
    
    df = pd.DataFrame(data, columns=['polyline_index', 'unused', 'x', 'y'])
    df.to_csv(output_file, index=False)

def main(svg_file, output_file, num_points=500):
    # Load the SVG file
    paths, attributes = svg2paths(svg_file)

    # Convert each path to polylines
    polylines = [path_to_polylines(path, num_points) for path in paths]

    # Save the polylines to a CSV file
    save_polylines_to_csv(polylines, output_file)

if __name__ == "__main__":
    # Input SVG file
    svg_file = r'output.svg'
    
    # Output CSV file for polylines
    output_file = 'polylines.csv'

    # Convert SVG paths to polylines and save to CSV
    main(svg_file, output_file)
