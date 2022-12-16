import random

line_count = 500
treshold = 1838

with open("synsets_with_glosses.txt") as synsets:
    synsets_lines = synsets.readlines()

with open("hyponyms_with_glosses.txt") as hyponyms:
    hyponyms_lines = hyponyms.readlines()

all_lines = synsets_lines + hyponyms_lines
print(len(set(all_lines)))

lins = set()
filenames = []
data = {}
max_treshold = 0
for i in range(1, 11):
    filenames.append(f"file_{i}.txt")

while len(set(lins)) < treshold:
    if len(set(lins)) > max_treshold:
        max_treshold = len(set(lins))
        print(max_treshold)
    written_lines = set()
    for i in range(1, 11):
        data[i] = set()
        # sample 500 lines from the combined list
        sampled_lines = random.choices(population=all_lines, k=line_count)
        # track the number of synsets and hyponyms lines in the sample
        synsets_count = 0
        hyponyms_count = 0

        # write the sampled lines to the file
        for line in sampled_lines:
            if line in synsets_lines:
                synsets_count += 1
            else:
                hyponyms_count += 1

            data[i].add(line)
            written_lines.add(line)

        # check if the proportion of synsets and hyponyms lines is within the required range
        # if not, continue sampling lines until it is
        while not (0.5 <= synsets_count / (synsets_count + hyponyms_count) <= 0.7) and \
                not (0.3 <= synsets_count / (synsets_count + hyponyms_count) <= 0.5):
            # sample an additional line from the remaining lines (not written to any file yet)
            additional_line = random.sample(set(all_lines) - written_lines, 1)[0]

            # write the additional line to the file
            data[i].add(additional_line)
            written_lines.add(additional_line)

            # update the count of synsets and hyponyms lines
            if additional_line in synsets_lines:
                synsets_count += 1
            else:
                hyponyms_count += 1
    lins = set()
    for i in range(1, 11):
        lins = lins.union(data[i])

for i in range(1, 11):
    with open(filenames[i - 1], "w") as f:
        f.writelines(data[i])
