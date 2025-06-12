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

1. Clone the WF repo:
```
git clone git@github.com:geosteering-no/DISTINGUISH-WF.git
```
3. Clone a fork of PET (not the main one): 
```
git clone git@github.com:KriFos1/PET.git kriPET
```
3. Change branch to `pipt-structure`
Now the custom PET is in kriPET folder. Let's assume one above the main code.
5. Follow the normal steps:
6. Make a virtual environment and install the two packages by standard "pip install -e ."
7. Install the custom PET, e.g.
```
pip install -e ../kriPET
```
9. Go to `DISTINGUISH_WD/data`
10. Run
```
python write_data.py
```
10. Go to `DISTINGUISH_WD/wf_demo`
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
