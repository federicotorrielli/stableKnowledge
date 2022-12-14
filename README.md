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
conda install -c conda-forge diffusers transformers accelerate ftfy tqdm scipy
```

### Image interrogator module

```bash
conda create -n Interrogation
conda activate Interrogation
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install .
```

### Image evaluation module

You can use the StableDiffusion conda environment.