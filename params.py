#   [mode]
#   Simulation name (should be supported by rur)
#       "hagn":     Horizon-AGN
#       "y01605":   YZiCS 01605
#       "y04466":   YZiCS 04466
#       ...
#       "y49096":   YZiCS 49096
#       "fornax":   FORNAX
#       "nh":       NewHorizon
#       "nh2":      NewHorizon2
#       "nc":       NewCluster
# mode = "y07206"
#   [ncpu]
#   Set nthread in numba
#   If ncpu <=0, skip
ncpu = 32

#   [galaxy]
#   Type of data
#       True:     Use galaxy and star
#       False:    Use halo and DM
galaxy = True

#   [nsnap]
#   How many snapshots to use for each galaxy
nsnap = 5

#   [overwrite]
#   If tree results already exist, overwrite or not?
#       True: overwrite
#       False: skip
overwrite = True

#   [logprefix]
#   file name of ytree log (./logprefix_iout.log)
# logprefix = f"ytree_"
logprefix = f"ytree_"

#   [detail]
#   Detail debugging in log
#       True:   DEBUG level
#       False:  INFO level
detail = True

#   [flush_GB]
#   Memory threshold for auto-flush in Gigabytes
flush_GB = 200


