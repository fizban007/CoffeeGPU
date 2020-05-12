#!/usr/bin/env python

import h5py
import numpy as np
#import pytoml
import toml
from pathlib import Path
import sys
import os
import re

def load_conf(path):
  conf_path = os.path.join(path, "config.toml")
  return toml.load(conf_path)
  #with open(conf_path, "rb") as f:
    #conf = pytoml.load(f)
    #conf = toml.load(f)
    #return conf

def xmf_head():
  return """<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf>
<Domain>
<Grid Name="Aperture" GridType="Collection" CollectionType="Temporal" >
  """

def xmf_step_header(nx, ny, t):
  return """<Grid Name="quadmesh" Type="Uniform">
  <Time Type="Single" Value="{2}"/>
  <Topology Type="2DSMesh" NumberOfElements="{0} {1}"/>
  <Geometry GeometryType="X_Y">
    <DataItem Dimensions="{0} {1}" NumberType="Float" Precision="4" Format="HDF">
      grid.h5:x1
    </DataItem>
    <DataItem Dimensions="{0} {1}" NumberType="Float" Precision="4" Format="HDF">
      grid.h5:x2
    </DataItem>
  </Geometry>
  """.format(nx, ny, t)

def xmf_step_close():
  return """</Grid>
"""

def xmf_tail():
  return """</Grid>
</Domain>
</Xdmf>
"""

def xmf_field_entry(name, step, nx, ny):
  return """<Attribute Name="{0}" Center="Node" AttributeType="Scalar">
    <DataItem Dimensions="{2} {3}" NumberType="Float" Precision="4" Format="HDF">
      fld.{1:05d}.h5:{0}
    </DataItem>
  </Attribute>
  """.format(name, step, nx, ny)

if len(sys.argv) < 2:
  print("Please specify path of the data!")
  sys.exit(1)

path = sys.argv[1]
conf = load_conf(path)
print(conf['dt'])
nx = conf['N'][0] // conf['downsample']
ny = conf['N'][1] // conf['downsample']
dx = conf['size'][0] / nx
dy = conf['size'][1] / ny
lower_x = conf['lower'][0]
lower_y = conf['lower'][1]

# Generate a grid hdf5 file
f_grid = h5py.File(os.path.join(path, "grid.h5"), "w")
x1 = np.array((ny, nx))
x2 = np.array((ny, nx))
r = np.exp(np.linspace(0.0, conf['size'][0], nx) + lower_x)
th = np.linspace(0.0, conf['size'][1], ny) + lower_y
rgrid, thgrid = np.meshgrid(r, th)
x1 = rgrid * np.sin(thgrid)
x2 = rgrid * np.cos(thgrid)
f_grid['x1'] = x1
f_grid['x2'] = x2
print(rgrid.shape)
f_grid.close()

# Generate a xmf file

# Generate a list of output steps
num_re = re.compile(r"\d+")
fld_steps = [
  int(num_re.findall(f.stem)[0]) for f in Path(path).glob("fld.*.h5")
]
fld_steps.sort()
# print(fld_steps)
# print(xmf_head())
# print(xmf_step_header(nx, ny))
# print(xmf_step_close())
# print(xmf_tail())
if len(fld_steps) > 0:
  f_fld = h5py.File(os.path.join(path, f"fld.{fld_steps[0]:05d}.h5"), "r")
  fld_keys = list(f_fld.keys())
  f_fld.close()
  print(fld_keys)
  with open(os.path.join(path, 'data.xmf'), "w") as output:
    output.write(xmf_head())
    for n in range(len(fld_steps)):
      step = fld_steps[n]
      time = step * conf['data_interval'] * conf['dt']

      output.write(xmf_step_header(ny, nx, time))
      for k in fld_keys:
        output.write(xmf_field_entry(k, step, ny, nx))
      output.write(xmf_step_close())
    output.write(xmf_tail())
