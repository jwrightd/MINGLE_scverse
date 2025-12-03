import MINGLE as mg
import sys
from pathlib import Path
import inspect
print("MINGLE imported OK")
print("version:", getattr(mg, "__version__", None))

# --- 1. See what submodules exist ---
print("\nSubmodules on MINGLE:")
print("tl:", mg.tl)
print("pl:", mg.pl)
print("pp:", mg.pp)

# --- 2. List functions inside each submodule using dir() ---
print("\nFunctions in tl:")
print([name for name in dir(mg.tl) if not name.startswith("_")])

print("\nFunctions in pl:")
print([name for name in dir(mg.pl) if not name.startswith("_")])

print("\nFunctions in pp:")
print([name for name in dir(mg.pp) if not name.startswith("_")])

# --- 3. Inspect a specific function (example: centroid_Calculation) ---
if hasattr(mg.tl, "centroid_Calculation"):
    print("\nSignature of tl.centroid_Calculation:")
    print(inspect.signature(mg.tl.centroid_Calculation))

    print("\nDocstring of tl.centroid_Calculation:")
    print(inspect.getdoc(mg.tl.centroid_Calculation))
else:
    print("\nmg.tl.centroid_Calculation not found – check tl/__init__.py exports.")