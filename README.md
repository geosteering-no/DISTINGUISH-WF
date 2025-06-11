## Installation
1. Install in a virtual environment (verified with python 3.11)
```
python -m venv .venv
source .venv/bin/activate
pip install -e .
```
 
2. There are currently issues with automatic instalation of PET data-assimilation library. To install it manually clone and install PET from Kristian's repo
```
cd ..
git clone git@github.com:KriFos1/PET.git
```

Change the branch
```
cd PET
git checkout pipt-structure
```

Install PET:
```
pip install -e .
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
