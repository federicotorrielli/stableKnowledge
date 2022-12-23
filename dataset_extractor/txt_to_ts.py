with open(f"dataset.txt") as file:
    lines = file.readlines()
    # Remove every \n from the lines
    lines = [line.replace("\n", "") for line in lines]
    with open(f"data.ts", "w") as data_file:
        data_file.write(f"export const data = {lines}")

print("Done")
