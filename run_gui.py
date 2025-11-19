"""
Launcher for GUI (avoids using python -m)
"""

import runpy
import os
import sys

# Force project root as working directory
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Run the GUI module
runpy.run_module("gui.main_app", run_name="__main__", alter_sys=True)
