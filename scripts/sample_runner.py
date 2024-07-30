from helper_for_csvs import read_csv, plot

csv_path = r'C:\Users\GURDARSH VIRK\OneDrive\Documents\adobe-gensolve-2024\problems\problems\frag0.csv'
paths = read_csv(csv_path)

print(paths)
# print(len(paths))
plot(paths)