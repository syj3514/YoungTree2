from rur import uri, uhmi
import numpy as np
import sys
import os
import time
import gc
from collections.abc import Iterable
import inspect
import psutil
import logging
import copy
import traceback
from numpy.lib.recfunctions import append_fields

# from tree_utool import *
if "/home/jeon/YoungTree2" in sys.path:
    sys.path.append("/home/jeon/YoungTree2")
from ytool import *
from yleaf import Leaf

class Treebase():
    __slots__ = ['iniGB', 'iniMB', 'flush_GB', 'simmode', 'galaxy', 'logprefix', 'detail',
                'partstr', 'Partstr', 'galstr', 'Galstr','verbose', 'debugger',
                'rurmode', 'repo',
                'nout','nstep','dp', 'part_halo_match',
                'dict_snap','dict_part','dict_gals','dict_leaves', 'branches_queue','resultdir']
    def __init__(self, simmode='hagn', galaxy=True, flush_GB=50, verbose=2, debugger=None, prefix="", prog=True, logprefix="output_", detail=True, dp=False, resultdir=None):
        func = f"[__Treebase__]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=verbose, debugger=debugger)

        self.iniGB = GB()
        self.iniMB = MB()
        self.flush_GB = flush_GB
        self.simmode = simmode
        self.galaxy = galaxy
        self.logprefix=logprefix
        self.detail=detail
        self.dp=dp
        self.resultdir=resultdir

        if self.galaxy:
            self.partstr = "star"
            self.Partstr = "Star"
            self.galstr = "gal"
            self.Galstr = "GalaxyMaker"
        else:
            self.partstr = "dm"
            self.Partstr = "DM"
            self.galstr = "halo"
            self.Galstr = "HaloMaker"
        self.verbose = verbose
        self.debugger = debugger

        self.repo, self.rurmode, _ = mode2repo(simmode)

        self.nout = load_nout(mode=self.simmode, galaxy=self.galaxy)
        self.nstep = load_nstep(mode=self.simmode, galaxy=self.galaxy, nout=self.nout)
        self.dict_snap = {} # in {iout}, RamsesSnapshot object
        self.dict_part = {} # in {iout}, in {galid}, Particle object
        self.dict_gals = {"galaxymakers":{}, "gmpids":{}, "galids":{}} # in each key, in {iout}, Recarray obejct
        self.dict_leaves = {} # in {iout}, in {galid}, Leaf object
        self.part_halo_match = {}
        gc.collect()

        clock.done()

    def summary(self, isprint=False):
        temp = [f"{key}({sys.getsizeof(self.dict_snap[key].part_data) / 2**20:.2f} MB) | " for key in self.dict_snap.keys()]
        tsnap = "".join(temp)
        
        temp = []
        for key in self.dict_part.keys():
            idict = self.dict_part[key]
            keys = list(idict.keys())
            temp += f"\t{key}: {len(idict)} {self.galstr}s with {np.sum([len(idict[ia]['id']) for ia in keys])} {self.Partstr}s\n"
        tpart = "".join(temp)

        temp = []
        for key in self.dict_gals["galaxymakers"].keys():
            if key in self.dict_gals["gmpids"].keys():
                temp += f"\t{key}: {len(self.dict_gals['galaxymakers'][key])} {self.galstr}s with {np.sum([len(ia) for ia in self.dict_gals['gmpids'][key]])} {self.Partstr}s\n"
            else:
                temp += f"\t{key}: {len(self.dict_gals['galaxymakers'][key])} {self.galstr}s with 0 {self.Partstr}s\n"
        tgm = "".join(temp)

        temp = []
        for key in self.dict_leaves.keys():
            temp += f"\t{key}: {len(self.dict_leaves[key])} leaves\n"
        tleaf = "".join(temp)
        
        text = f"\n[Tree Data Report]\n\n>>> Snapshot\n{tsnap}\n>>> {self.Partstr}\n{tpart}>>> {self.Galstr}\n{tgm}>>> Leaves\n{tleaf}\n>>> Used Memory: {psutil.Process().memory_info().rss / 2 ** 30:.4f} GB\n"

        if isprint:
            print(text)
        return text
    
    def load_snap(self, iout:int, prefix:str="")->uri.RamsesSnapshot:
        # if psutil.Process().memory_info().rss / 2 ** 30 > self.flush_GB+self.iniGB:
        #     self.flush_auto(prefix=prefix)

        if not iout in self.dict_snap.keys():
            func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}({iout}) "
            clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)
            self.dict_snap[iout] = uri.RamsesSnapshot(self.repo, iout, mode=self.rurmode, path_in_repo='snapshots')
            clock.done()
        if not iout in self.dict_part.keys():
            self.dict_part[iout] = {}

        return self.dict_snap[iout]
    
    def load_gals(self, iout:int, galid=None, return_part=False, prefix=""):
        # if psutil.Process().memory_info().rss / 2 ** 30 > self.flush_GB+self.iniGB:
        #     self.flush_auto(prefix=prefix)
        if galid is None:
            galid = 'all'

        # Save
        if return_part:
            if not iout in self.dict_gals["gmpids"].keys():
                func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}({iout}) "
                clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

                snap = self.load_snap(iout, prefix=prefix)
                gm, gmpid = uhmi.HaloMaker.load(snap, galaxy=self.galaxy, load_parts=True, double_precision=self.dp) #<-- Bottleneck!
                self.dict_gals["galaxymakers"][iout] = gm
                cumparts = np.insert(np.cumsum(gm["nparts"]), 0, 0)
                gmpid = tuple(gmpid[ cumparts[i]:cumparts[i+1] ] for i in range(len(gm)))
                self.dict_gals["gmpids"][iout] = gmpid #<-- Bottleneck!
                del gm; del gmpid; del cumparts

                clock.done()
        else:
            if not iout in self.dict_gals["galaxymakers"].keys():
                func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}({iout}) "
                clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

                snap = self.load_snap(iout, prefix=prefix)
                gm = uhmi.HaloMaker.load(snap, galaxy=self.galaxy, load_parts=False, double_precision=self.dp) #<-- Bottleneck!
                self.dict_gals["galaxymakers"][iout] = gm
                del gm

                clock.done()

        # Load
        if isinstance(galid,str):           # 'all'
            if galid=='all':
                if return_part:
                    b = tuple(self.dict_gals["gmpids"][iout][ia-1] for ia in self.dict_gals["galaxymakers"][iout]['id'])
                    return self.dict_gals["galaxymakers"][iout], b
                return self.dict_gals["galaxymakers"][iout]
        elif isinstance(galid, Iterable):   # Several galaxies
            a = np.hstack([self.dict_gals["galaxymakers"][iout][ia-1] for ia in galid])
            if return_part:
                b = tuple(self.dict_gals["gmpids"][iout][ia-1] for ia in galid)
                return a, b
            return a
        else:                               # One galaxy
            if return_part:
                return self.dict_gals["galaxymakers"][iout][galid-1], self.dict_gals["gmpids"][iout][galid-1]
            return self.dict_gals["galaxymakers"][iout][galid-1]
    
    def load_part(self, iout:int, galid=None, prefix="")->uri.RamsesSnapshot.Particle:
        # if psutil.Process().memory_info().rss / 2 ** 30 > self.flush_GB+self.iniGB:
        #     self.flush_auto(prefix=prefix)
        if galid is None:
            galid = 'all'

        # Save
        snap = self.load_snap(iout, prefix=prefix)
        if galid == 'all':
            galids = self.load_gals(iout, galid='all', return_part=False)['id']
            calc=False
            for iid in galids:
                if not iid in self.dict_part[iout].keys():
                    calc=True
                    break
            if calc:
                func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}({iout} all) "
                clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

                snap.box = np.array([[0, 1], [0, 1], [0, 1]])
                snap.get_part(pname=self.partstr, target_fields=['id', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'm'])
                snap.part_data['id'] = np.abs(snap.part_data['id'])
                arg = np.argsort(snap.part_data['id'])
                snap.part_data = snap.part_data[arg]
                gals, gpids = self.load_gals(iout, galid=None, return_part=True)
                gpids_flat = np.concatenate(gpids)
                haloids = np.repeat(gals['id'], gals['nparts'])
                self.part_halo_match[iout] = np.zeros(len(snap.part_data), dtype=int)
                self.part_halo_match[iout][gpids_flat-1] = haloids
                # snap.part_data = append_fields(snap.part_data, "haloids", temp, usemask=False)
                snap.part.table = snap.part_data
                self.dict_snap[iout] = snap

                # gals, gpids = self.load_gals(iout, galid=None, return_part=True)
                for gal, gpid in zip(gals, gpids):
                    if not gal['id'] in self.dict_part[iout].keys():
                        part = snap.part[gpid-1]
                        if not hasattr(part, 'table'):
                            part = snap.Particle(part, snap)
                        self.dict_part[iout][gal['id']] = part
                    elif len(self.dict_part[iout]) != gal['npart']:
                        part = snap.part[gpid-1]
                        if not hasattr(part, 'table'):
                            part = snap.Particle(part, snap)
                        self.dict_part[iout][gal['id']] = part
                    else:
                        pass
                
                clock.done()
            return self.dict_snap[iout].part

        else:
            if not galid in self.dict_part[iout].keys():
                func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}({iout}) "
                clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

                cpulist = snap.get_halos_cpulist( np.atleast_1d(self.load_gals(iout, galid=galid)), radius=1.5, radius_name='r' )
                snap.get_part(pname=self.partstr, target_fields=['id', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'm'], cpulist=cpulist)
                snap.part_data['id'] = np.abs(snap.part_data['id'])
                self.dict_snap[iout] = snap

                gal, gpid = self.load_gals(iout, galid=galid, return_part=True)
                part = snap.part[large_isind(snap.part['id'], gpid)]
                if not hasattr(part, 'table'):
                    part = snap.Particle(part, snap)
                self.dict_part[iout][gal['id']] = part
            
                clock.done()
            return self.dict_part[iout][galid]

    def load_leaf(self, iout:int, galid:int, backup=None, prefix="") -> Leaf:
        # func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}({iout}) "
        # clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        if not iout in self.dict_leaves.keys():
            self.dict_leaves[iout] = {}
        
        if not galid in self.dict_leaves[iout].keys():
            if backup is not None:
                gal = None
                part = None
            else:
                gal = self.load_gals(iout, galid, return_part=False, prefix=prefix)
                part = self.load_part(iout, galid, prefix=prefix)
            self.dict_leaves[iout][galid] = Leaf(gal, part, self.nout, verbose=self.verbose-1, prefix=prefix, debugger=self.debugger, backup=backup)
        
        # clock.done()
        return self.dict_leaves[iout][galid] 
        
    def update_debugger(self, iout=None):
        if iout is None:
            for iout in self.dict_leaves.keys():
                for jkey in self.dict_leaves[iout].keys():
                    self.dict_leaves[iout][jkey].debugger = self.debugger
        else:
            for jkey in self.dict_leaves[iout].keys():
                self.dict_leaves[iout][jkey].debugger = self.debugger
    
    def flush(self, iout:int, prefix="", leafclear=False):
        keys = list(self.dict_snap.keys())
        if iout in keys:
            self.dict_snap[iout].clear()
            del self.dict_snap[iout]
        
        keys = list(self.dict_gals['galaxymakers'].keys())
        if iout in keys:
            self.dict_gals['galaxymakers'][iout] = []
            del self.dict_gals['galaxymakers'][iout]
        
        keys = list(self.dict_gals['gmpids'].keys())
        if iout in keys:
            self.dict_gals['gmpids'][iout] = []
            del self.dict_gals['gmpids'][iout]
        
        keys = list(self.dict_part.keys())
        if iout in keys:
            keys2 = list(self.dict_part[iout].keys())
            for key in keys2:
                self.dict_part[iout][key].snap.clear()
                self.dict_part[iout][key].table = []
                del self.dict_part[iout][key]
            del self.dict_part[iout]
        
        if leafclear:
            keys = list(self.dict_leaves.keys())
            if iout in keys:
                func = f"[{inspect.stack()[0][3]}]"; prefix1 = f"{prefix}{func}({iout}) "
                clock = timer(text=prefix1, verbose=self.verbose, debugger=self.debugger)

                keys2 = list(self.dict_leaves[iout].keys())
                for key in keys2:
                    self.dict_leaves[iout][key].clear()
                    del self.dict_leaves[iout][key]
                del self.dict_leaves[iout]
                
                clock.done()

            keys = list(self.part_halo_match.keys())
            if iout in keys:
                self.part_halo_match[key] = []
                del self.part_halo_match[key]
        gc.collect()

        
        
        
        
        
        
        
        
        # def flush
    # def flush auto