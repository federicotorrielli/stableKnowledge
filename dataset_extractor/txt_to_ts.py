# Given files from file_1.txt to file_10.txt (in the same directory as the script), each one containing multiple lines,
# Create files data_1.ts to data_10.ts each one containing a single line like this:
# export const data_n = ["line1", "line2", ...] (where n is the number of the file)

for i in range(1, 11):
    with open(f"file_{i}.txt") as file:
        lines = file.readlines()
        # Remove every \n from the lines
        lines = [line.replace("\n", "") for line in lines]
        with open(f"data_{i}.ts", "w") as data_file:
            data_file.write(f"export const data_{i} = {lines}")
