#!/usr/bin/env python3

import h5py
import numpy as np
import toml
from pathlib import Path
import os
import re

class Data:
  def __init__(self, path=None):
   if path is not None:
      self.open_data(path)

  def __dir__(self):
    return (
      self._fld_keys
      + self._ptc_keys
      + ["load", "load_fld", "load_ptc", "keys", "conf"]
    )

  def __getattr__(self, key):
    if key not in self.__dict__:
      if key in self._fld_keys:
        self.__load_fld_quantity(key)
      elif key in self._ptc_keys:
        self.__load_ptc_quantity(key)
      # elif key in self._mesh_keys:
      #   self.__load_mesh_quantity(key)
      elif key == "keys":
        self.__dict__[key] = self._fld_keys + self._ptc_keys# + self._mesh_keys
      elif key == "conf":
        self.__dict__[key] = self._conf
      else:
        return None
    return self.__dict__[key]

  def __load_fld_quantity(self, key):
    path = os.path.join(self._path, f"fld.{self._current_fld_step:05d}.h5")
    data = h5py.File(path, "r")
    self.__dict__[key] = data[key][()]
    data.close()

  def __load_ptc_quantity(self, key):
    pass

  # def __load_mesh_quantity(self, key):
  #   data = h5py.File(self._meshfile, "r")
  #   self.__dict__[key] = data[key][()]
  #   data.close()

  def open_data(self, path):
    self._conf = self.load_conf(os.path.join(path, "config.toml"))
    self._path = path

    # find mesh deltas
    self.delta = np.zeros(len(self._conf["N"]))
    for n in range(len(self.delta)):
      self.delta[n] = self._conf["size"][n] / self._conf["N"][n] * self._conf["downsample"]

    num_re = re.compile(r"\d+")
    # generate a list of output steps for fields
    self._fld_keys = []
    self.fld_steps = [
      int(num_re.findall(f.stem)[0]) for f in Path(path).glob("fld.*.h5")
    ]
    if len(self.fld_steps) > 0:
      self.fld_steps.sort()
      self._current_fld_step = self.fld_steps[0]
      f_fld = h5py.File(
        os.path.join(self._path, f"fld.{self._current_fld_step:05d}.h5"),
        "r",
      )
      self._fld_keys = list(f_fld.keys()) + ["B", "J", "flux"]
      f_fld.close()

    # generate a list of output steps for particles
    self._ptc_keys = []
    self.ptc_steps = [
      int(num_re.findall(f.stem)[0]) for f in Path(path).glob("ptc.*.h5")
    ]
    if len(self.ptc_steps) > 0:
      self.ptc_steps.sort()
      self._current_ptc_step = self.ptc_steps[0]
      f_ptc = h5py.File(
        os.path.join(self._path, f"ptc.{self._current_ptc_step:05d}.h5"),
        "r",
      )
      self._ptc_keys = list(f_ptc.keys())
      f_ptc.close()

  def unload(self):
    self.unload_fld()
    self.unload_ptc()

  def unload_fld(self):
    for k in self._fld_keys:
      if k in self.__dict__:
        self.__dict__.pop(k, None)
        # self._mesh_loaded = False

  def unload_ptc(self):
    for k in self._ptc_keys:
      if k in self.__dict__:
        self.__dict__.pop(k, None)

  def load(self, step):
    self.load_fld(step)
    self.load_ptc(step)

  def load_fld(self, step):
    if not step in self.fld_steps:
      print("Field step not in data directory!")
      return
    self._current_fld_step = step
    self.unload_fld()

  def load_ptc(self, step):
    if not step in self.ptc_steps:
      print("Ptc step not in data directory!")
      return
    self._current_ptc_step = step
    self.unload_ptc()

  def load_conf(self, path):
    return toml.load(path)
