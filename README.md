# Distinguish Workflow Demo

## Installation

**Currently, some dependencies are being updated to enhance performance. Follow the Temporary workaround for installation**
1. Install in a virtual environment (verified with python 3.11)
```
python -m venv .venv
source .venv/bin/activate
pip install -e .
```
### Work-around for a special PET version
2. Clone a fork of PET (not the main one): 
```
git clone git@github.com:KriFos1/PET.git kriPET
```
3. Change branch to `pipt-structure`
```
git checkout -b pipt-structure origin/pipt-structure
```

Now the custom PET is in kriPET folder. Let's assume one above the main code.
### Continue installation normally

4. Install the custom PET in the DISTINGUISH-WF folder (activating the venv if not already), e.g.
```
cd ../DISTINGUISH-WF
source .venv/bin/activate
```
```
pip install -e ../kriPET
```

## Running
Make some initial data by

```
cd data
python write_data.py
cd ../
```

Run by typing in the terminal:

```
cd wf_demo
run_WF
```

## Training data available

For GAN training:
- https://github.com/geosteering-no/geosteering-gan-training-2020

For EM training:
- https://zenodo.org/records/17776294

## Cite as

Introduced in:
- Alyaev, S., Fossum, K., Djecta, H. E., Tveranger, J., & Elsheikh, A. H. (2024). **DISTINGUISH Workflow: a New Paradigm of Dynamic Well Placement Using Generative Machine Learning**. *In EAGE ECMOR 2024*, 2024. https://doi.org/10.3997/2214-4609.202437018. (arXiv: https://arxiv.org/abs/2503.08509) 

The article is under peer review with the name "A Generative-AI Modeling Framework for Explainable Decision Support in Complex
Geosteering Scenarios"

### BibTeX

```
@inproceedings{Alyaev2024ECMORDISTINGUISH,
  author    = {Alyaev, Sergey and Fossum, Kristian and Djecta, Hibat Errahmen and Tveranger, Jan and Elsheikh, Ahmed H.},
  title     = {DISTINGUISH Workflow: a New Paradigm of Dynamic Well Placement Using Generative Machine Learning},
  booktitle = {ECMOR 2024 (EAGE), Conference Proceedings},
  year      = {2024},
  doi       = {10.3997/2214-4609.202437018},
  url       = {https://arxiv.org/abs/2503.08509}
}
```
