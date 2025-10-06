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
Now the custom PET is in kriPET folder. Let's assume one above the main code.
### Continue installation normally

4. Install the custom PET, e.g.
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

Run by typing in terminal:

```
cd wf_demo
run_WF
```
