import matplotlib.pyplot as plt
import pandas as pd

# Read abs_concr.txt
with open('abs_concr.txt', 'r') as f:
    abs_concr_data = {}
    for line in f.readlines():
        word, score = line.strip().lower().split()
        abs_concr_data[word] = int(score)

# Read basicness_scores.xlsx
basicness_scores = pd.read_excel('basicness_scores.xlsx', engine='openpyxl')

# Process the data
plot_data = []
for index, row in basicness_scores.iterrows():
    term = row[0].split(',')[0].lower()
    basicness_score = row[1]

    if term in abs_concr_data:
        abs_concr_score = abs_concr_data[term]
        plot_data.append((term, basicness_score, abs_concr_score))

# Plot the data
fig, ax = plt.subplots()
for term, basicness_score, abs_concr_score in plot_data:
    ax.scatter(basicness_score, abs_concr_score, color='blue')

ax.set_xlabel('Basicness Score (0-1.0)')
ax.set_ylabel('Concreteness Score (0-700)')
ax.set_title('Basicness Score vs Concreteness Score')

plt.show()
