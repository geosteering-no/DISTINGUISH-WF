## Installation

**Currently, some dependencies are being updated to enhance performance. Follow the Temporary workaround for installation**
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

### Temporary workaround due to ongoing upgrade of dependencies

1. Clone the WF repo: git@github.com:geosteering-no/DISTINGUISH-WF.git
2. Clone a fork of PET (not the main one): 
```
git clone git@github.com:KriFos1/PET.git
```
3. Change branch to `pipt-structure`
4. Follow the normal steps:
5. Make a virtual environment and install the two packages by standard "pip install -e ."
6. Go to `DISTINGUISH_WD/data`
7. Run
```
python write_data.py
```
8. Go to `DISTINGUISH_WD/wf_demo`
Run by writing:
```
run_WF
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
