#!/usr/bin/env python3

import h5py
import numpy as np
import sys
import os
from datalib_logsph import Data

path = sys.argv[1]
data = Data(path)
conf = data._conf

for n in data.fld_steps:
  data.load(n)
  f = h5py.File(os.path.join(path, f"fld.{n:05d}.h5"))
  f["flux"] = data.flux
  f.close()
  print(n)
