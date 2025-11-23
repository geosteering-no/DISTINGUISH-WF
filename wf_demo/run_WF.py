import matplotlib
matplotlib.use("Agg")

import subprocess
import os
import sys

def main():
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    # Launch Streamlit with the app
    subprocess.run(["streamlit", "run", app_path] + sys.argv[1:])

# USAGE
# To run the Streamlit app, execute this script from the command line:
# run_WF --server.port 8888