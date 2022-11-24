# Thesis pipeline

## Installation

Generating and interrogating images MUST be split in two different envs.
Creating only one for two will cause dependency problems.

### Image generator module

```bash
conda create -n StableDiffusion
conda activate StableDiffusion
conda install -c conda-forge diffusers transformers accelerate ftfy tqdm
conda install -c "xformers/label/dev" xformers
```

### Image interrogator module

```bash
python3 -m venv ci_env
source ci_env/bin/activate
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
pip install -e git+https://github.com/pharmapsychotic/BLIP.git@lib#egg=blip
pip install clip-interrogator
```