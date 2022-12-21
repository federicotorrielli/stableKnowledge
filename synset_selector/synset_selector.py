import sys
# Given two files "synsets.txt" (or take it from argv, if there is) and "other_synsets.txt", this script
# Makes the user choose to flush in a file called "selected_synsets.txt" and "other_selected_synsets.txt"
# the synsets that presents in the terminal with an input. Pressing enter will flush the synset in the file, don't flush it otherwise.
# The script will stop when the user press "q" or "Q" in the terminal or when the number of synsets flushed is 500.
# Every line in the two files corresponds to a synset. When a synset from synsets.txt is flushed, the corresponding
# synset in other_synsets.txt at the same exact line in the file will be flushed too.

synsets_to_flush = set() # keep track of the synsets that the user wants to flush
other_synsets_to_flush = set()
synsets = {} # fill this with key (synsets.txt line): value (other_synsets.txt line, at the same line number)

if len(sys.argv) > 1:
    synsets_file = str(sys.argv[1])
else:
    synsets_file = "synsets.txt"

if len(sys.argv) > 2:
    other_synsets_file = str(sys.argv[2])
else:
    other_synsets_file = "other_synsets.txt"

# read the two files
with open(synsets_file, "r") as f:
    synsets_lines = f.readlines()
with open(other_synsets_file, "r") as f:
    other_synsets_lines = f.readlines()

# fill the synsets dictionary
for i in range(len(synsets_lines)):
    synsets[i] = other_synsets_lines[i]

# flush the synsets in the terminal
for i in range(len(synsets_lines)):
    print(f"Synset {i}: {synsets_lines[i]}")
    print(f"Corresponding synset {i}: {other_synsets_lines[i]}")
    user_input = input("Flush this synset? (y/n): ")
    if user_input == "y" or user_input == "Y" or user_input == "":
        synsets_to_flush.add(synsets_lines[i])
        other_synsets_to_flush.add(synsets[i])
        print("-" * 20)

