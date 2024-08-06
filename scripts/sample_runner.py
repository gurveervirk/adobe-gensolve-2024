from helper_for_csvs import read_csv, plot

csv_path = r'problems\problems\isolated.csv'
paths = read_csv(csv_path)

# print(paths)
# print(len(paths))
plot(paths)