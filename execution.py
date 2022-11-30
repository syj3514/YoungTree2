import time
from rur import uri
from YoungTree3.ytool import *
from YoungTree3.yroot import Treebase
from YoungTree3.ymain import do_onestep
import importlib
import sys

if len(sys.argv) < 3:
    params = importlib.import_module("params")
else:
    params = sys.argv[2]
    if params[-3:] == '.py':
        params = importlib.import_module(params[:-3])
#########################################################
#   From params.py, record to dictionary
#########################################################
p = {}
for key in params.__dict__.keys():
    if not "_" in key:
        p[key] = params.__dict__[key]
    p["flush_GB"] = params.flush_GB
p = DotDict(p)

#########################################################
#   Load nout
#########################################################
modenames = {"hagn": "Horizon-AGN", 
            "y01605": "YZiCS-01605",
            "y04466": "YZiCS-04466",
            "y05420": "YZiCS-05420",
            "y05427": "YZiCS-05427",
            "y06098": "YZiCS-06098",
            "y07206": "YZiCS-07206",
            "y10002": "YZiCS-10002",
            "y17891": "YZiCS-17891",
            "y24954": "YZiCS-24954",
            "y29172": "YZiCS-29172",
            "y29176": "YZiCS-29176",
            "y35663": "YZiCS-35663",
            "y36413": "YZiCS-36413",
            "y36415": "YZiCS-36415",
            "y39990": "YZiCS-39990",
            "y49096": "YZiCS-49096",
            "nh": "NewHorizon",
            "nh2": "NewHorizon2",
            "nc": "NewCluster",
            "fornax": "FORNAX",
            "custom": "Custom"
            }

#########################################################
#   Mode Configuration
#########################################################

if len(sys.argv) < 2:
    mode = input("mode=? ")
else:
    mode = sys.argv[1]

if not mode in modenames.keys():
    raise ValueError(f"`{mode}` is not supported!\nSee {list(modenames.keys())}")
modename = modenames[mode]
repo, rurmode, dp = mode2repo(mode)
if dp:
    if not p.galaxy:
        dp = False

# For message printing
galstr = "Halo"
galstrs = "Halos"
if p.galaxy:
    galstr = "Galaxy"
    galstrs = "Galaxies"

# Read output list
nout = load_nout(mode=mode, galaxy=p.galaxy)
nstep = load_nstep(mode=mode, galaxy=p.galaxy, nout=nout)

#########################################################
#   Make debugger for log
#########################################################
# Initialize
debugger = None
inidebugger = None
resultdir = f"{repo}/YoungTree/"
if not os.path.isdir(resultdir):
    os.mkdir(resultdir)
fname = make_logname(mode, -1, logprefix=p.logprefix, dirname=resultdir)
# repo/YoungTree/ytree_ini.log
inidebugger = custom_debugger(fname, detail=p.detail)
inihandlers = inidebugger.handlers
message = f"< YoungTree >\nUsing {modename} {galstr}\n{len(nout)} outputs are found! ({nout[-1]}~{nout[0]})\n\nSee `{fname}`\n\n"
inidebugger.info(message)
print(message)

#########################################################
###############         Data base                  ######
#########################################################
inidebugger.info(f"\nAllow {p.flush_GB:.2f} GB Memory")
print(f"Allow {p.flush_GB:.2f} GB Memory")

if p.ncpu>0:
    from numba import set_num_threads
    set_num_threads(p.ncpu)
    import ytool
    ytool.ncpu = p.ncpu

MyTree = Treebase(simmode=mode, galaxy=p.galaxy, debugger=inidebugger, verbose=0, flush_GB=p.flush_GB, detail=p.detail, logprefix=p.logprefix, dp=dp, resultdir=resultdir, )


#########################################################
###############         Execution                  ######
#########################################################
uri.timer.verbose=0
reftot = time.time()
for iout in nout:
    main = do_onestep(MyTree,iout, p, inidebugger, reftot=reftot, mode=mode, nout=nout, nstep=nstep, resultdir=resultdir)

outs = list(MyTree.dict_leaves.keys())
for out in outs:
    MyTree.flush(out, leafclear=True)        
    main.reducebackup_debug(out)

dprint_("\nDone\n", inidebugger)
print("\nDone\n")
