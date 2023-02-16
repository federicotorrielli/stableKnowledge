# stableKnowledge

A synset-to-image-to-description custom pipeline using Stable Diffusion and BLIP
to discover new frontiers in the world of Natural Language Processing!

## Installation

Generating, Interrogating and Evaluating images MUST be split in three different envs.
Creating only one for the three will cause dependency problems.

### Image Generator module

```bash
conda create -n Generation
conda activate Generation
conda install -c conda-forge diffusers transformers safetensors accelerate tqdm xformers
python3 pipeline.py generate
```

### Image Interrogator module

```bash
conda create -n Interrogator
conda activate Interrogator
pip install -U salesforce-lavis spacy pillow
python3 -m spacy download en_core_web_sm
python3 pipeline.py interrogate
```

### Image Evaluation module

```bash
conda create -n Evaluator
conda activate Evaluator
conda install -c conda-forge sentence-transformers scipy matplotlib
python3 pipeline.py evaluate
```

## Usage

### Term Extraction

In the folder `term_extractor` you can find the script `extract.py` that generates the raw list of terms to be used in
the pipeline: this file uses the Transformer Filter class.
The common nouns from Ogden are in the file `ogden.txt` and the terms from Reddit are in the file `common_nouns.txt`.
The _Transformer Filter_ class is in the file `transformer_filter.py` and serves the purpose of creating
*OPT-basic/advanced* words. Don't expect the same exact result as our experimentation, words are randomly extract at the
end of the process. Given that this process is very time consuming, we decided to provide the final list of words in the
files `synsets.txt` and `hyponyms.txt` in the main folder: these are, respectively, the OPT-basic/advanced words.

### Term Refinement

The dataset of 500 terms is created by using the scripts contained in the folder `synset_selector`.

### Pipeline

The pipeline is composed of three main steps: image generation, image interrogation and image evaluation.
The script `pipeline.py` is the main script that orchestrates the whole process. First, you have to enter in the
correct environment and then you can run the script with the command `python3 pipeline.py <command>` where the command
can be one of the following: `generate`, `interrogate` and `evaluate`. The rest is self-explanatory.

## Code Explanation

> The following code explanations have been automatically generated

### abstract_concrete_identifier.py

This Python code takes a text file containing synsets as input, and it generates an output file that assigns each synset
to one of two classes: 'concrete' or 'abstract'. The classification is done using logistic regression on word vectors
obtained from the Spacy library.

The first step is to import the required packages and download a pre-trained model from Spacy. Next, the code reads the
input file, which is a text file containing synsets, where each synset is enclosed in parentheses and contains a string
in quotes. The code then uses the NLTK WordNet library to extract the synsets and stores them in a list.

The code then defines two classes, 'concrete' and 'abstract', and creates a training set for the logistic regression
classifier. The training set consists of two lists of words, one for each class. The word vectors are obtained from the
Spacy model, and the X and y matrices are created using NumPy. The classifier is trained using logistic regression with
a balanced class weight and a regularization parameter C of 0.1.

The code then iterates over each synset and assigns it to a class based on the logistic regression classifier's
prediction. The synset name and class are written to an output file called 'dataset_with_classes.txt'.

Finally, the code reads another input file, 'super_annotator.txt', which contains synsets, and assigns each synset to
the 'concrete' class if it is predicted as such by the classifier. These synsets are written to an output file called '
super_annotator_concrete.txt'.

In summary, this code uses a logistic regression classifier to classify synsets as either 'concrete' or 'abstract' based
on word vectors obtained from the Spacy library. The output is written to two text files: 'dataset_with_classes.txt'
and 'super_annotator_concrete.txt'.

### abstract_vs_concrete_evaluation.py

This code defines a set of functions to calculate and plot the agreement between the concreteness of words and their
cosine similarity scores. Here is how the code works:

The code begins by importing the required modules and libraries such as os, pickle, matplotlib, numpy, sklearn.metrics,
and tqdm.

Next, there are three functions defined in this code:

1. `get_concreteness_from_file(file_name: str)`: Given a file containing a word list, the function returns a dictionary
   of
   words with their concreteness integer values. The function reads the text file line by line and splits each line into
   a word and its corresponding concreteness score. The function then stores these words and scores as key-value pairs
   in a dictionary and returns the dictionary.
2. `get_cossim_score_from_folders(folder_path: str)`: Given a folder containing subfolders, each of which contains a
   text
   file with cosine similarity scores for various words, the function returns a dictionary of words with their mean
   cosine similarity scores. The function first reads the cosine scores file for each subfolder, takes the first five
   lines of the file, converts the scores into floats, and calculates their mean. Then, it extracts the first word in
   the subfolder name that does not contain an underscore and adds the word and its corresponding mean cosine similarity
   score to a dictionary. The function then returns the dictionary.
3. `calculate_agreement(concreteness_dict, cosine_dict, concreteness_threshold, cosine_threshold, report=False)`: Given
   a
   dictionary of words and their concreteness scores, a dictionary of words and their cosine similarity scores, and
   thresholds for concreteness and cosine similarity scores, the function returns the Cohen's kappa coefficient between
   the two sets of scores. The function first creates two dictionaries of word labels (concrete or abstract) for the
   given concreteness and cosine similarity scores based on the specified thresholds. It then generates a list of these
   labels for each word, and finally returns the Cohen's kappa coefficient between the two sets of labels. If report is
   True, the function returns a classification report instead of the kappa coefficient.

Then, the code defines another function called find_correct_threshold(concreteness_dict: dict, cosine_dict: dict) that
searches for the optimal concreteness and cosine similarity thresholds that maximize the agreement between the two sets
of scores. The function does this by iterating through a range of values for the two thresholds and computing the
agreement between the two sets of scores for each combination of values. The function returns the optimal thresholds and
the corresponding agreement between the two sets of scores.

Finally, the code defines a function plot_agreements(agreements_with_thresholds: list[tuple]) that takes a list of
tuples containing the agreements and corresponding threshold values and plots them on a heatmap. The x-axis of the plot
represents the cosine similarity threshold, while the y-axis represents the concreteness threshold. The color of each
point on the heatmap represents the corresponding agreement between the concreteness and cosine similarity scores for
the given thresholds.

### json_analyzer.py

This is a collection of functions that are used for analyzing annotation data, specifically for a task that requires
labeling images with two possible labels, "middle" or "advanced".

`list_shared_items(coders: list) -> list`: This function takes a list of coders and returns a list of the items for
which
all the coders gave the same answer of "advanced". The function first creates an empty dictionary to store the answers
for each item. Then it iterates through the tuples in coders, which consist of (coder, item, answer), and adds the
answers to the dictionary. If the item is not already in the dictionary, it creates a list for it. Finally, the function
checks if all the answers for each item in the dictionary are True and appends the item to shared_items if they are.

`calculate_agreement(data: list[dict]) -> None`: This function takes a list of dictionaries data, which contain
information about the labels given by individual coders, and calculates the level of agreement between the coders. The
function first truncates the arrays to the lowest of them if they are different in size, then creates a list of tuples (
coder, item, label) by iterating through each item in the dataset for each coder. The AnnotationTask object is created
using this list of tuples, and then the function prints the kappa, fleiss, alpha, and scotts scores.

`calculate_agreement_sliding_window(data: list[dict], window_size=130) -> None`: This function is similar to
calculate_agreement, but it calculates agreement using a sliding window of size window_size. It first truncates the
arrays to the lowest of them if they are different in size, then creates a list of tuples (coder, item, label) by
iterating through each item in the dataset for each coder. The function then creates n AnnotationTasks, each with a
window size of window_size, and prints the kappa, fleiss, alpha, and scotts scores for each task.

`calculate_coherence_index(data: list[dict])`: This function calculates the coherence index for each coder. The
coherence
index measures the extent to which a coder's answers are consistent with their previous answers for the same item. The
function iterates through each coder's data and calculates the coherence index, then prints the result.

`calculate_hard_probability(data: list[dict]) -> None`: This function calculates the probability of each item being
labeled "middle" or "advanced" if it is a hard question. It first creates two lists, hard_middle_answers and
hard_advanced_answers, which contain the answers for each hard question. It then calculates the probability of each
answer being "middle" or "advanced" and prints the results.

`common_hard_answers(data: list[dict], n=10) -> None`: This function finds the common answers for the hard questions and
associates each answer to the synset in the dataset. It first creates a list of all the answers to hard questions, then
associates each answer with the synset in the dataset. The function then counts the number of times each answer appears
and prints the n most common answers.