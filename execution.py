import time
from rur import uri
from YoungTree3.ytool import *
from YoungTree3.yroot import Treebase
from YoungTree3.ymain import do_onestep
import importlib
import sys

print("\n$ python3 execution.py <MODE> <PARAMS.py>\n")

#########################################################
#   From params.py, record to dictionary
#########################################################
if len(sys.argv) < 3:
    print("PARAM are not given, use `params.py`")
    params = importlib.import_module("params")
else:
    params = sys.argv[2]
    if params[-3:] == '.py':
        print(f"Read `{params}`")
        params = importlib.import_module(params[:-3])
p = {}
for key in params.__dict__.keys():
    if not "_" in key:
        p[key] = params.__dict__[key]
    p["flush_GB"] = params.flush_GB
p = DotDict(p)




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


#########################################################
#   Load nout
#########################################################
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
# `repo/YoungTree/ytree_ini.log`
inidebugger = custom_debugger(fname, detail=p.detail)
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
