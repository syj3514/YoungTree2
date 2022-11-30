import numpy as np

default = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('vx', 'f8'), ('vy', 'f8'), ('vz', 'f8'), ('m', 'f8')]
part_dtype = default + [('id', 'i4'), ('level', 'i4'), ('cpu', 'i4')]
hydro_dtype = ['rho', 'vx', 'vy', 'vz', 'P']

repo = "/storage2/jeon/from_tardis/gem_50_8"
rurmode = 'dm_only'
dp = False
