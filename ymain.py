import gc
import psutil
import time
import numpy as np
from rur import uri, uhmi
import warnings
import logging
from ytool import *
from yroot import Treebase
from yleaf import Leaf
import importlib
import inspect
import traceback

params = importlib.import_module("params")
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
            "fornax": "FORNAX"
            }

# Mode configuration
import sys

mode = sys.argv[1]

if len(sys.argv) < 2:
    mode = input("mode=? ")

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
debugger = None
inidebugger = None
# Initialize
resultdir = f"{repo}/YoungTree/"
if not os.path.isdir(resultdir):
    os.mkdir(resultdir)
fname = make_logname(mode, -1, logprefix=p.logprefix)
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

MyTree = Treebase(simmode=mode, galaxy=p.galaxy, debugger=inidebugger, verbose=0, flush_GB=p.flush_GB, detail=p.detail, logprefix=p.logprefix, dp=dp, resultdir=resultdir)


#########################################################
###############         Tree Building              ######
#########################################################
def loadout(Tree:Treebase, iout:int, resultdir=None):
    func = f"[{inspect.stack()[0][3]}]"; prefix = f"{func}({iout}) "
    clock = timer(text=prefix, verbose=Tree.verbose, debugger=Tree.debugger)

    backups=None
    if os.path.isfile(f"{resultdir}ytree_{iout:05d}_temp.pickle"):
        dprint_(f"{prefix} load from `{resultdir}ytree_{iout:05d}_temp.pickle`", Tree.debugger)
        backups, _ = pklload(f"{resultdir}ytree_{iout:05d}_temp.pickle")
    Tree.load_snap(iout, prefix=prefix)
    if backups is None:
        Tree.load_gals(iout, galid='all', return_part=True, prefix=prefix)
        Tree.load_part(iout, galid='all', prefix=prefix)
    else:
        Tree.load_gals(iout, galid='all', return_part=False, prefix=prefix)
    clock2 = timer(text=f"{prefix}[load_leaf]", verbose=Tree.verbose, debugger=Tree.debugger)
    for galid in Tree.dict_gals['galaxymakers'][iout]['id']:
        backup = None
        if (backups is not None):
            if galid in backups.keys():
                backup = backups[galid]
        Tree.load_leaf(iout, galid, backup=backup, prefix=prefix)
    clock2.done(add=f"({len(Tree.dict_gals['galaxymakers'][iout]['id'])} gals)")
    Tree.flush(iout)

    clock.done()

def find_cands(Tree:Treebase, iout:int, jout:int, mcut=0.01, resultdir=None, prefix=""):
    func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}({iout}<->{jout}) "
    clock = timer(text=prefix, verbose=Tree.verbose, debugger=Tree.debugger)

    keys = list(Tree.dict_leaves[iout].keys())
    jhalos = None
    # backups = {}
    for key in keys:
        leaf:Leaf = Tree.dict_leaves[iout][key]
        calc = True
        if(leaf.prog is not None):
            if(jout in leaf.prog[:,0]):
                calc=False
        if(leaf.desc is not None):
            if(jout in leaf.desc[:,0]):
                calc=False
        
        if calc:
            if jhalos is None:
                try:
                    jhalos = Tree.part_halo_match[jout]
                except:
                    _, jhalos = pklload(f"{resultdir}ytree_{jout:05d}_temp.pickle")
            pid = leaf.pid
            pid = pid[pid <= len(jhalos)]
            hosts = jhalos[pid-1]
            hosts = hosts[hosts>0]
            hosts, count = np.unique(hosts, return_counts=True)
            hosts = hosts[count/len(pid) > mcut]
            if len(hosts)>0:
                otherleaves = [Tree.load_leaf(jout, iid) for iid in hosts]
                ids, scores = leaf.calc_score(jout, otherleaves)
            else:
                ids = np.array([[jout, 0]])
                scores = np.array([[-10, -10, -10, -10, -10]])

            if jout<iout:
                leaf.prog = ids if leaf.prog is None else np.vstack((leaf.prog, ids))
                leaf.prog_score = scores if leaf.prog_score is None else np.vstack((leaf.prog_score, scores))
                leaf.changed = True
            elif jout>iout:
                leaf.desc = ids if leaf.desc is None else np.vstack((leaf.desc, ids))
                leaf.desc_score = scores if leaf.desc_score is None else np.vstack((leaf.desc_score, scores))
                leaf.changed = True
            else:
                raise ValueError(f"Same output {iout} and {jout}!")
        
        else:
            if jout<iout:
                arg = leaf.prog[:,0]==jout
                ids = leaf.prog[arg]
                scores = leaf.prog_score[arg]
            elif jout>iout:
                arg = leaf.desc[:,0]==jout
                ids = leaf.desc[arg]
                scores = leaf.desc_score[arg]
            else:
                raise ValueError(f"Same output {iout} and {jout}!")
            
        # backups[key] = leaf.selfsave()
        msg = f"{prefix}<{leaf.name()}> has {len(ids)} candidates"
        if len(ids)>0:
            if np.sum(scores[0])>0:
                if len(ids) < 6:
                    msg = f"{msg} {[f'{ids[i][-1]}({scores[i][0]:.3f})' for i in range(len(ids))]}"
                else:
                    msg = f"{msg} {[f'{ids[i][-1]}({scores[i][0]:.3f})' for i in range(5)]+['...']}"
            else:
                msg = f"{prefix}<{leaf.name()}> has {len(ids)-1} candidates"
        dprint_(msg, Tree.debugger)
    leaf = None
    # pklsave(backups, f"{resultdir}ytree_{iout:05d}.pickle", overwrite=True)

    clock.done()

def LEAFbackup(Tree:Treebase, resultdir=None):
    iouts = list(Tree.dict_leaves.keys())
    for iout in iouts:
        prefix = f"[LEAFbackup]({iout})"
        clock = timer(text=prefix, verbose=Tree.verbose, debugger=Tree.debugger)
        keys = list(Tree.dict_leaves[iout].keys())
        backups = {}
        if os.path.isfile(f"{resultdir}ytree_{iout:05d}_temp.pickle"):
            backups, parthalomatch = pklload(f"{resultdir}ytree_{iout:05d}_temp.pickle")
            dprint_(f"{prefix} Overwrite `{resultdir}ytree_{iout:05d}_temp.pickle`", Tree.debugger)
        else:
            parthalomatch = Tree.part_halo_match[iout]
        for key in keys:
            leaf:Leaf = Tree.dict_leaves[iout][key]
            if leaf.changed:
                backups[key] = leaf.selfsave()
        leaf = None
        pklsave((backups, parthalomatch), f"{resultdir}ytree_{iout:05d}_temp.pickle", overwrite=True)
    
        clock.done()

def reducebackup(Tree:Treebase, iout:int, resultdir=None):
    prefix = f"[Reduce Backup file] ({iout})"
    clock = timer(text=prefix, verbose=Tree.verbose, debugger=Tree.debugger)

    if not os.path.isfile(f"{resultdir}ytree_{iout:05d}_temp.pickle"):
        raise FileNotFoundError(f"`{resultdir}ytree_{iout:05d}_temp.pickle` is not found!")
    file, _ = pklload(f"{resultdir}ytree_{iout:05d}_temp.pickle")
    if isinstance(file, dict):
        keys = list(file.keys())
        count = 0
        for key in keys:
            gals = file[key]['gal'] if count==0 else np.hstack((gals, file[key]['gal']))
            count += 1
        pklsave(gals, f"{resultdir}ytree_{iout:05d}.pickle")
        dprint_(f"{prefix} Save `{resultdir}ytree_{iout:05d}.pickle`", Tree.debugger)
        os.remove(f"{resultdir}ytree_{iout:05d}_temp.pickle")
        dprint_(f"{prefix} Remove `{resultdir}ytree_{iout:05d}_temp.pickle`", Tree.debugger)
    clock.done()

def treerecord(iout, nout, elapse_s, total_elapse_s, debugger:logging.Logger):
    a = f"{iout} done ({elapse_s/60:.2f} min elapsed)"
    dprint_(a, debugger)
    aver = total_elapse_s/60/len(nout[nout>=iout])
    a = f"{len(nout[nout>=iout])}/{len(nout)} done ({aver:.2f} min/snap)"
    dprint_(a, debugger)
    a = f"{aver*len(nout[nout<iout]):.2f} min forecast"
    dprint_(a, debugger)
    a = f"{psutil.Process().memory_info().rss / 2 ** 30:.4f} GB used\n" # memory used
    dprint_(a, debugger)

uri.timer.verbose=0
reftot = time.time()
for iout in nout:
    try:
        ref = time.time()
        skip = False
        backups = None
        if os.path.isfile(f"{resultdir}ytree_{iout:05d}.pickle"):
            backups = pklload(f"{resultdir}ytree_{iout:05d}.pickle")
            if not p.overwrite:
                skip=True
        if not skip:
            # New log file
            fname = make_logname(MyTree.simmode, iout, logprefix=MyTree.logprefix)
            # MyTree.debugger.handlers = []
            MyTree.debugger = custom_debugger(fname, detail=MyTree.detail)
            # Load snap gal part
            dprint_(f"\n\nStart at iout={iout}\n", MyTree.debugger, level='info')
            loadout(MyTree, iout, resultdir=resultdir)
            istep = out2step(iout, galaxy=p.galaxy, mode=mode, nout=nout, nstep=nstep)
            for j in range(p.nsnap):
                jstep = istep-j-1
                if jstep > 0:
                    jout = step2out(jstep, galaxy=p.galaxy, mode=mode, nout=nout, nstep=nstep)
                    dprint_(f"\n\nProgenitor at jout={jout}\n", MyTree.debugger, level='info')
                    loadout(MyTree, jout, resultdir=resultdir)
                    find_cands(MyTree, iout, jout, resultdir=resultdir)
            for j in range(p.nsnap):
                jstep = istep+j+1
                if jstep <= np.max(nstep):
                    jout = step2out(jstep, galaxy=p.galaxy, mode=mode, nout=nout, nstep=nstep)
                    if not jout in MyTree.dict_snap.keys():
                        dprint_(f"\n\nDescendant at jout={jout}\n", MyTree.debugger, level='info')
                        loadout(MyTree, jout, resultdir=resultdir)
                        find_cands(MyTree, iout, jout, resultdir=resultdir)
            
            cutstep = istep+5
            if cutstep<=np.max(nstep):
                cutout = step2out(cutstep, galaxy=p.galaxy, mode=mode, nout=nout, nstep=nstep)
                outs = list(MyTree.dict_leaves.keys())
                for out in outs:
                    if out > cutout:
                        MyTree.flush(out, leafclear=True)        
                        reducebackup(MyTree, out, resultdir=resultdir)
            
            LEAFbackup(MyTree, resultdir=resultdir)
            MyTree.debugger.info(f"\n{MyTree.summary()}\n")
            treerecord(iout, nout, time.time()-ref, time.time()-reftot, inidebugger)
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        MyTree.debugger.error(traceback.format_exc())
        MyTree.debugger.error(e)
        MyTree.debugger.error(MyTree.summary())
        raise ValueError("Iteration is terminated")