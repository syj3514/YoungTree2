# YoungTree support several modes:
#   "hagn" (Horizon-AGN)
#   "y01605" (YZiCS-01605)
#   "y04466" (YZiCS-04466)
#   "y05420" (YZiCS-05420)
#   "y05427" (YZiCS-05427)
#   "y06098" (YZiCS-06098)
#   "y07206" (YZiCS-07206)
#   "y10002" (YZiCS-10002)
#   "y17891" (YZiCS-17891)
#   "y24954" (YZiCS-24954)
#   "y29172" (YZiCS-29172)
#   "y29176" (YZiCS-29176)
#   "y35663" (YZiCS-35663)
#   "y36413" (YZiCS-36413)
#   "y36415" (YZiCS-36415)
#   "y39990" (YZiCS-39990)
#   "y49096" (YZiCS-49096)
#   "nh" (NewHorizon)
#   "nh2" (NewHorizon2)
#   "nc" (NewCluster)
#   "fornax" (FORNAX)

# If you want to use custom mode,
# set mode="custom", then YoungTree will use below parameters

default = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('vx', 'f8'), ('vy', 'f8'), ('vz', 'f8'), ('m', 'f8')]

# part_dtype:
#   Particle column names and dtypes
part_dtype = default + [('id', 'i4'), ('level', 'i4'), ('cpu', 'i4')]

# hydro dtype:
#   Hydro column names and dtypes
hydro_dtype = ['rho', 'vx', 'vy', 'vz', 'P']

# repo
#   Repository which should have `snapshots` and `halo` or `galaxy` directory
repo = "/storage2/jeon/from_tardis/gem_100_8"

# rurmode
#   mode name which can be supported in rur package
rurmode = 'dm_only'

# dp
#   Whether double_precision is used or not in GalaxyMaker or HaloMaker
dp = False
