import MINGLE as mg
import sys
from pathlib import Path
import inspect
print("MINGLE imported OK")
print("version:", getattr(mg, "__version__", None))

#test run
mg.tl.centroid_Calculation()