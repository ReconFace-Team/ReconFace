# run_gui.py
import os
import sys

# ---- Forzar proyecto raíz en sys.path ----
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import runpy
runpy.run_module("gui.main_app", run_name="__main__", alter_sys=True)
