with open(f"data.ts") as data_file:
    line = data_file.readline()
    line = line.replace("export const data = ", "")
    # Now we are left with a big ["string", "string", ...]
    # We want to have a list of strings
    # First we remove the first and last character
    line = line[1:-1]
    # Now we split the string by the comma
    lines = line.split("\",")
    # Now we remove the " + space from the start of each string
    # With the exception of the first string
    lines[0] = lines[0][1:]
    # And for the other strings, we remove the " + space from the start
    for i in range(1, len(lines)):
        lines[i] = lines[i][2:]
        # If we are on the last string, we remove the last character
        if i == len(lines) - 1:
            lines[i] = lines[i][:-1]

# Time to save the data
with open(f"dataset2.txt", "w") as file:
    for line in lines:
        file.write(f"{line}\n")
"{line}\n")
