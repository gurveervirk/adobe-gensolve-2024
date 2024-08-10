from helper import read_csv, plot_simple

filename = r'misc-outputs\polylines.csv'
paths = read_csv(filename)
plot_simple(paths, filename.split('\\')[-1].split('.')[0] + '.png')