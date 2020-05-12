#!/usr/bin/env python3

import h5py
import numpy as np
import toml
from pathlib import Path
import os
import re


class Data:
  _coord_keys = ["x1", "x2", "r", "theta", "rv", "thetav", "dr", "dtheta"]

  def __init__(self, path):
    conf = self.load_conf(path)
    self._conf = conf
    self._path = path

    # Load mesh file
    self._meshfile = os.path.join(path, "grid.h5")

    # Generate a list of output steps
    num_re = re.compile(r"\d+")
    self.fld_steps = [
      int(num_re.findall(f.stem)[0]) for f in Path(path).glob("fld.*.h5")
    ]
    if len(self.fld_steps) == 0:
      raise(ValueError("No field output files found!"))
    self.fld_steps.sort()

    self._current_fld_step = self.fld_steps[0]
    self._mesh_loaded = False

    f_fld = h5py.File(
      os.path.join(self._path, f"fld.{self._current_fld_step:05d}.h5"),
      "r"
    )
    self._fld_keys = list(f_fld.keys()) + ["B", "EdotB", "flux", "J", "JdotB", "Br", "Bth", "Bph",
                                           "Er", "Eth", "Eph", "dBr", "dBth", "dBph", "dEr",
                                           "dEth", "dEph", "Br0", "Bth0", "Bph0", "Er0", "Eth0", "Eph0",
                                           "Jr", "Jth", "Jph", "B2", "E2", "U"]
    self._Bx0 = f_fld['Bx'][()]
    self._By0 = f_fld['By'][()]
    self._Bz0 = f_fld['Bz'][()]
    self._Ex0 = f_fld['Ex'][()]
    self._Ey0 = f_fld['Ey'][()]
    self._Ez0 = f_fld['Ez'][()]
    self._bg_keys = ["Bx0", "By0", "Bz0", "Ex0", "Ey0", "Ez0"]
    f_fld.close()
    self.__dict__.update(("_" + k, None) for k in self._fld_keys)
    self._load_mesh()

  def __dir__(self):
    return (
      self._fld_keys
      + self._coord_keys
      + ["load", "load_fld", "keys", "conf"]
    )

  def __getattr__(self, key):
    if key in (self._fld_keys + self._coord_keys + self._bg_keys):
      content = getattr(self, "_" + key)
      if content is not None:
        return content
      else:
        self._load_content(key)
        return getattr(self, "_" + key)
    elif key == "keys":
      return self._fld_keys
    elif key == "conf":
      return self._conf
    else:
      return None

  def load(self, step):
    self.load_fld(step)

  def load_fld(self, step):
    if not step in self.fld_steps:
      print("Field step not in data directory!")
      return
    self._current_fld_step = step
    for k in self._fld_keys:
      if k not in Data._coord_keys:
        setattr(self, "_" + k, None)
        # self._mesh_loaded = False

  def _load_content(self, key):
    if key in self._fld_keys:
      self._load_fld(key)
    elif key in self._coord_keys:
      self._load_mesh()

  def _load_fld(self, key):
    path = os.path.join(self._path, f"fld.{self._current_fld_step:05d}.h5")
    data = h5py.File(path, "r")
    if key == "flux":
      self._load_mesh()
      dtheta = (
        self._theta[self._conf["guard"][1] + 2]
        - self._theta[self._conf["guard"][1] + 1]
      )
      self._flux = np.cumsum(
        self.Br * self._rv * self._rv * np.sin(self._thetav) * dtheta, axis=0
      )
    elif key == "B":
      self._B = np.sqrt(self.Br * self.Br + self.Bth * self.Bth + self.Bph * self.Bph)
    elif key == "J":
      self._J = np.sqrt(self.Jr * self.Jr + self.Jth * self.Jth + self.Jph * self.Jph)
    elif key == "EdotB":
      self._EdotB = self.Er * self.Br + self.Eth * self.Bth + self.Eph * self.Bph
    elif key == "JdotB":
      self._JdotB = self.Jr * self.Br + self.Jth * self.Bth + self.Jph * self.Bph
    elif key == "Br0":
      self._Br0 = self._rv * self._Bx0
    elif key == "Bth0":
      self._Bth0 = self._rv * self._By0
    elif key == "Bph0":
      self._Bph0 = self._rv * np.sin(self._thetav) * self._Bz0
    elif key == "Er0":
      self._Er0 = self._rv * self._Ex0
    elif key == "Eth0":
      self._Eth0 = self._rv * self._Ey0
    elif key == "Eph0":
      self._Eph0 = self._rv * np.sin(self._thetav) * self._Ez0
    elif key == "dBr":
      self._dBr = self._rv * (self.Bx - self._Bx0)
    elif key == "dBth":
      self._dBth = self._rv * (self.By - self._By0)
    elif key == "dBph":
      self._dBph = self._rv * np.sin(self._thetav) * (self.Bz - self._Bz0)
    elif key == "dEr":
      self._dEr = self._rv * (self.Ex - self._Ex0)
    elif key == "dEth":
      self._dEth = self._rv * (self.Ey - self._Ey0)
    elif key == "dEph":
      self._dEph = self._rv * np.sin(self._thetav) * (self.Ez - self._Ez0)
    elif key == "Br":
      self._Br = self._rv * self.Bx
    elif key == "Bth":
      self._Bth = self._rv * self.By
    elif key == "Bph":
      self._Bph = self._rv * np.sin(self._thetav) * self.Bz
    elif key == "Er":
      self._Er = self._rv * self.Ex
    elif key == "Eth":
      self._Eth = self._rv * self.Ey
    elif key == "Eph":
      self._Eph = self._rv * np.sin(self._thetav) * self.Ez
    elif key == "Jr":
      self._Jr = self._rv * self.Jx
    elif key == "Jth":
      self._Jth = self._rv * self.Jy
    elif key == "Jph":
      self._Jph = self._rv * np.sin(self._thetav) * self.Jz
    elif key == "B2":
      self._B2 = self.Br**2 + self.Bth**2 + self.Bph**2
    elif key == "E2":
      self._E2 = self.Er**2 + self.Eth**2 + self.Eph**2
    elif key == "U":
      self._U = (self.B2 + self.E2)/2.0
      # elif key == "EdotB":
      #     setattr(self, "_" + key, data["EdotBavg"][()])
    else:
      setattr(self, "_" + key, data[key][()])
      data.close()

  def _load_mesh(self):
    if self._mesh_loaded:
      return
    meshfile = h5py.File(self._meshfile, "r")

    self._x1 = meshfile["x1"][()]
    self._x2 = meshfile["x2"][()]
    # self._r = np.pad(
    #     np.exp(
    #         np.linspace(
    #             0,
    #             self._conf["Grid"]["size"][0],
    #             self._conf["Grid"]["N"][0]
    #             // self._conf["Simulation"]["downsample"],
    #         )
    #         + self._conf["Grid"]["lower"][0]
    #     ),
    #     self._conf["Grid"]["guard"][0],
    #     "constant",
    # )
    # self._theta = np.pad(
    #     np.linspace(
    #         0,
    #         self._conf["Grid"]["size"][1],
    #         self._conf["Grid"]["N"][1] // self._conf["Simulation"]["downsample"],
    #     )
    #     + self._conf["Grid"]["lower"][1],
    #     self._conf["Grid"]["guard"][1],
    #     "constant",
    # )
    self._r = np.exp(
      np.linspace(
        0,
        self._conf["size"][0],
        self._conf["N"][0]
        // self._conf["downsample"],
      ) + self._conf["lower"][0]
    )
    self._theta = np.linspace(
      0,
      self._conf["size"][1],
      self._conf["N"][1] // self._conf["downsample"],
    ) + self._conf["lower"][1]

    meshfile.close()
    self._rv, self._thetav = np.meshgrid(self._r, self._theta)
    self._dr = (
      self._r[self._conf["guard"][0] + 2]
      - self._r[self._conf["guard"][0] + 1]
    )
    self._dtheta = (
      self._theta[self._conf["guard"][1] + 2]
      - self._theta[self._conf["guard"][1] + 1]
    )

    self._mesh_loaded = True

  def load_conf(self, path):
    conf_path = os.path.join(path, "config.toml")
    return toml.load(conf_path)
    # with open(conf_path, "rb") as f:
    #   conf = pytoml.load(f)
    #   return conf
