# stableKnowledge

A synset-to-image-to-description custom pipeline using Stable Diffusion, CLIP and BLIP
to discover new frontiers in the world of Natural Language Processing!

## Installation

Generating and interrogating images MUST be split in two different envs.
Creating only one for the two will cause dependency problems.

### Image generator module

```bash
conda create -n StableDiffusion
conda activate StableDiffusion
conda install -c conda-forge diffusers transformers accelerate ftfy tqdm scipy
```

### Image interrogator module

```bash
python3 -m venv ci_env
source ci_env/bin/activate
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
pip install -e git+https://github.com/AdamOswald/BLIP.git@lib#egg=blip
pip install clip-interrogator
```

### Image evaluation module

You can use the StableDiffusion conda environment.