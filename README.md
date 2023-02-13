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