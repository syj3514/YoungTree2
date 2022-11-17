import psutil
import time
import numpy as np
import logging
from ytool import *
from yroot import Treebase
import inspect
import traceback


#########################################################
###############         Tree Building              ######
#########################################################
class Main:
    def __init__(self, Tree:Treebase, resultdir:str="./"):
        self.Tree = Tree
        self.resultdir = resultdir
        
    
        
    def loadout_debug(self, iout:int):
        Tree = self.Tree
        resultdir = self.resultdir
        @DebugDecorator
        def _loadout_debug(Tree:Treebase, iout:int, resultdir="./"):
            func = f"[{inspect.stack()[0][3]}]"; prefix = f"{func}({iout}) "

            backups=None
            if os.path.isfile(f"{resultdir}ytree_{iout:05d}_temp.pickle"):
                backups, _ = pklload(f"{resultdir}ytree_{iout:05d}_temp.pickle")
            Tree.load_snap(iout, prefix=prefix)
            if backups is None:
                Tree.load_gals(iout, galid='all', return_part=True, prefix=prefix)
                Tree.load_part(iout, galid='all', prefix=prefix)
            else:
                Tree.load_gals(iout, galid='all', return_part=False, prefix=prefix)
            for galid in Tree.dict_gals['galaxymakers'][iout]['id']:
                backup = None
                if (backups is not None):
                    if galid in backups.keys():
                        backup = backups[galid]
                Tree.load_leaf(iout, galid, backup=backup, prefix=prefix)
            Tree.flush(iout, prefix=prefix)
            backups = None; del backups
            backup = None; del backup
        return _loadout_debug(Tree, iout, resultdir=resultdir)
    


    def find_cands_debug(self, iout:int, jout:int):
        Tree = self.Tree
        resultdir = self.resultdir
        @DebugDecorator
        def _find_cands(Tree:Treebase, iout:int, jout:int, mcut=0.01, resultdir="./", prefix=""):
            func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}({iout}<->{jout}) "

            keys = list(Tree.dict_leaves[iout].keys())
            jhalos = None
            # backups = {}
            for key in keys:
                # Calc, or not?
                calc = True
                if(Tree.dict_leaves[iout][key].prog is not None):
                    if(jout in Tree.dict_leaves[iout][key].prog[:,0]):
                        calc=False
                if(Tree.dict_leaves[iout][key].desc is not None):
                    if(jout in Tree.dict_leaves[iout][key].desc[:,0]):
                        calc=False
                # Main calculation
                if calc:
                    if jhalos is None:
                        try:
                            jhalos = Tree.part_halo_match[jout]
                        except:
                            _, jhalos = pklload(f"{resultdir}ytree_{jout:05d}_temp.pickle")
                    pid = Tree.dict_leaves[iout][key].pid
                    pid = pid[pid <= len(jhalos)]
                    hosts = jhalos[pid-1]
                    hosts = hosts[hosts>0]
                    hosts, count = np.unique(hosts, return_counts=True) # CPU?
                    hosts = hosts[count/len(pid) > mcut]
                    if len(hosts)>0:
                        otherleaves = [Tree.load_leaf(jout, iid) for iid in hosts]
                        ids, scores = Tree.dict_leaves[iout][key].calc_score(jout, otherleaves) # CPU?
                    else:
                        ids = np.array([[jout, 0]])
                        scores = np.array([[-10, -10, -10, -10, -10]])
                    if jout<iout:
                        Tree.dict_leaves[iout][key].prog = ids if Tree.dict_leaves[iout][key].prog is None else np.vstack((Tree.dict_leaves[iout][key].prog, ids))
                        Tree.dict_leaves[iout][key].prog_score = scores if Tree.dict_leaves[iout][key].prog_score is None else np.vstack((Tree.dict_leaves[iout][key].prog_score, scores))
                        Tree.dict_leaves[iout][key].changed = True
                    elif jout>iout:
                        Tree.dict_leaves[iout][key].desc = ids if Tree.dict_leaves[iout][key].desc is None else np.vstack((Tree.dict_leaves[iout][key].desc, ids))
                        Tree.dict_leaves[iout][key].desc_score = scores if Tree.dict_leaves[iout][key].desc_score is None else np.vstack((Tree.dict_leaves[iout][key].desc_score, scores))
                        Tree.dict_leaves[iout][key].changed = True
                    else:
                        raise ValueError(f"Same output {iout} and {jout}!")
                # No need to calculation
                else:
                    if jout<iout:
                        arg = Tree.dict_leaves[iout][key].prog[:,0]==jout
                        ids = Tree.dict_leaves[iout][key].prog[arg]
                        scores = Tree.dict_leaves[iout][key].prog_score[arg]
                    elif jout>iout:
                        arg = Tree.dict_leaves[iout][key].desc[:,0]==jout
                        ids = Tree.dict_leaves[iout][key].desc[arg]
                        scores = Tree.dict_leaves[iout][key].desc_score[arg]
                    else:
                        raise ValueError(f"Same output {iout} and {jout}!")
                # Debugging message
                msg = f"{prefix}<{Tree.dict_leaves[iout][key].name()}> has {len(ids)} candidates"
                if len(ids)>0:
                    if np.sum(scores[0])>0:
                        if len(ids) < 6:
                            msg = f"{msg} {[f'{ids[i][-1]}({scores[i][0]:.3f})' for i in range(len(ids))]}"
                        else:
                            msg = f"{msg} {[f'{ids[i][-1]}({scores[i][0]:.3f})' for i in range(5)]+['...']}"
                    else:
                        msg = f"{prefix}<{Tree.dict_leaves[iout][key].name()}> has {len(ids)-1} candidates"
                dprint_(msg, Tree.debugger, level='debug')
        return _find_cands(Tree, iout, jout, resultdir=resultdir)

    

    def LEAFbackup_debug(self):
        Tree = self.Tree
        resultdir = self.resultdir
        @DebugDecorator
        def _LEAFbackup(Tree:Treebase, resultdir="./"):
            iouts = list(Tree.dict_leaves.keys())
            for iout in iouts:
                prefix = f"[LEAFbackup]({iout})"

                keys = list(Tree.dict_leaves[iout].keys())
                backups = {}
                if os.path.isfile(f"{resultdir}ytree_{iout:05d}_temp.pickle"):
                    backups, parthalomatch = pklload(f"{resultdir}ytree_{iout:05d}_temp.pickle")
                    dprint_(f"{prefix} Overwrite `{resultdir}ytree_{iout:05d}_temp.pickle`", Tree.debugger, level='info')
                else:
                    parthalomatch = Tree.part_halo_match[iout]
                for key in keys:
                    if Tree.dict_leaves[iout][key].changed:
                        backups[key] = Tree.dict_leaves[iout][key].selfsave()
                pklsave((backups, parthalomatch), f"{resultdir}ytree_{iout:05d}_temp.pickle", overwrite=True)
            del parthalomatch
            del backups
        return _LEAFbackup(Tree, resultdir=resultdir)

    

    def reducebackup_debug(self, iout:int):
        Tree = self.Tree
        resultdir = self.resultdir
        @DebugDecorator
        def _reducebackup(Tree:Treebase, iout:int, resultdir="./"):
            prefix = f"[Reduce Backup file] ({iout})"

            if not os.path.isfile(f"{resultdir}ytree_{iout:05d}_temp.pickle"):
                raise FileNotFoundError(f"`{resultdir}ytree_{iout:05d}_temp.pickle` is not found!")
            file, _ = pklload(f"{resultdir}ytree_{iout:05d}_temp.pickle")
            if isinstance(file, dict):
                keys = list(file.keys())
                count = 0
                for key in keys:
                    gals = file[key]['gal'] if count==0 else np.hstack((gals, file[key]['gal']))
                    count += 1
                pklsave(gals, f"{resultdir}ytree_{iout:05d}.pickle", overwrite=True)
                dprint_(f"{prefix} Save `{resultdir}ytree_{iout:05d}.pickle`", Tree.debugger, level='info')
                os.remove(f"{resultdir}ytree_{iout:05d}_temp.pickle")
                dprint_(f"{prefix} Remove `{resultdir}ytree_{iout:05d}_temp.pickle`", Tree.debugger, level='info')
            del gals
        return _reducebackup(Tree, iout, resultdir=resultdir)
  
    

def treerecord(iout:int, nout:int, elapse_s:float, total_elapse_s:float, debugger:logging.Logger):
    a = f"{iout} done ({elapse_s/60:.2f} min elapsed)"
    dprint_(a, debugger, level='info')
    aver = total_elapse_s/60/len(nout[nout>=iout])
    a = f"{len(nout[nout>=iout])}/{len(nout)} done ({aver:.2f} min/snap)"
    dprint_(a, debugger, level='info')
    a = f"{aver*len(nout[nout<iout]):.2f} min forecast"
    dprint_(a, debugger, level='info')
    a = f"{psutil.Process().memory_info().rss / 2 ** 30:.4f} GB used\n" # memory used
    dprint_(a, debugger, level='info')

    

def do_onestep(Tree:Treebase, iout:int,p:DotDict, inidebugger:logging.Logger, reftot=0, mode=None, nout=None, nstep=None, resultdir="./")->Main:
    if nout is None:
        nout = load_nout(mode=mode, galaxy=p.galaxy)
    if nstep is None:
        nstep = load_nstep(mode=mode, galaxy=p.galaxy, nout=nout)
    main = Main(Tree, resultdir)

    try:
        ref = time.time()
        skip = False
        
        # Fully saved
        if os.path.isfile(f"{resultdir}ytree_{iout:05d}.pickle"):
            dprint_(f"[Queue] {iout} is done --> Skip\n", inidebugger, level='info')
            skip=True
        
        # Temp exists
        if os.path.isfile(f"{resultdir}ytree_{iout:05d}_temp.pickle"):
            dprint_(f"[Queue] `{resultdir}ytree_{iout:05d}_temp.pickle` is found", inidebugger, level='info')
            istep = out2step(iout, galaxy=p.galaxy, mode=mode, nout=nout, nstep=nstep)
            cutstep = istep+5
            if cutstep <= np.max(nstep):
                cutout = step2out(cutstep, galaxy=p.galaxy, mode=mode, nout=nout, nstep=nstep)
                if os.path.isfile(f"{resultdir}ytree_{cutout:05d}_temp.pickle"):
                    dprint_(f"[Queue] `{resultdir}ytree_{cutout:05d}_temp.pickle` is found --> Do\n", inidebugger, level='info')
                    skip=False
                else:
                    dprint_(f"[Queue] `{resultdir}ytree_{cutout:05d}_temp.pickle` is not found --> Skip\n", inidebugger, level='info')
                    skip=True
            else:
                skip=False
    
        # Main process
        if not skip:
            # New log file
            dprint_(f"[Queue] {iout} start\n", inidebugger, level='info')
            fname = make_logname(Tree.simmode, iout, logprefix=Tree.logprefix)
            Tree.debugger = custom_debugger(fname, detail=Tree.detail)
            Tree.update_debugger()
            Tree.debugger.info(f"\n{Tree.summary()}\n")
            # Load snap gal part
            dprint_(f"\n\nStart at iout={iout}\n", Tree.debugger, level='info')
            main.loadout_debug(iout)
            istep = out2step(iout, galaxy=p.galaxy, mode=mode, nout=nout, nstep=nstep)
            # Find progenitors
            for j in range(p.nsnap):
                jstep = istep-j-1
                if jstep > 0:
                    jout = step2out(jstep, galaxy=p.galaxy, mode=mode, nout=nout, nstep=nstep)
                    dprint_(f"\n\nProgenitor at jout={jout}\n", Tree.debugger, level='info')
                    main.loadout_debug(jout)
                    main.find_cands_debug(iout, jout)
            # Find descendants
            for j in range(p.nsnap):
                jstep = istep+j+1
                if jstep <= np.max(nstep):
                    jout = step2out(jstep, galaxy=p.galaxy, mode=mode, nout=nout, nstep=nstep)
                    if not jout in Tree.dict_snap.keys():
                        dprint_(f"\n\nDescendant at jout={jout}\n", Tree.debugger, level='info')
                        main.loadout_debug(jout)
                        main.find_cands_debug(iout, jout)
            dprint_(f"\n\n", Tree.debugger, level='info')
            # Flush redundant snapshots
            cutstep = istep+5
            if cutstep<=np.max(nstep):
                cutout = step2out(cutstep, galaxy=p.galaxy, mode=mode, nout=nout, nstep=nstep)
                outs = list(Tree.dict_leaves.keys())
                for out in outs:
                    if out > cutout:
                        Tree.flush(out, leafclear=True)        
                        # reducebackup(Tree, out, resultdir=resultdir)
                        main.reducebackup_debug(out)
            # Backup files
            main.LEAFbackup_debug()
            Tree.debugger.info(f"\n{Tree.summary()}\n")
            treerecord(iout, nout, time.time()-ref, time.time()-reftot, inidebugger)
        return main
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        Tree.debugger.error(traceback.format_exc())
        Tree.debugger.error(e)
        Tree.debugger.error(Tree.summary())
        sys.exit("Iteration is terminated")
    return main