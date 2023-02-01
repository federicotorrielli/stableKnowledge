# stableKnowledge

A synset-to-image-to-description custom pipeline using Stable Diffusion and BLIP
to discover new frontiers in the world of Natural Language Processing!

## Installation

Generating and interrogating images MUST be split in two different envs.
Creating only one for the two will cause dependency problems.

### Image generator module

```bash
conda create -n Generation
conda activate Generation
conda install -c conda-forge diffusers transformers safetensors accelerate tqdm
python3 pipeline.py generate
```

### Image interrogator module

```bash
conda create -n Interrogator
conda activate Interrogator
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install -e .
pip install spacy pillow
python3 -m spacy download en_core_web_sm
python3 pipeline.py interrogate
```

### Image evaluation module

```bash
conda create -n Evaluator
conda activate Evaluator
conda install -c conda-forge sentence-transformers scipy matplotlib
python3 pipeline.py evaluate
```