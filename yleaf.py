from __future__ import annotations
import inspect
from numpy.lib.recfunctions import append_fields, drop_fields
from rur import uri
import copy

from ytool import *
from typing import List
# from yroot import Treebase


class Leaf():
    def __init__(self, gal, parts, nout, howmanystep=5, ncut=None, verbose=1, prefix="", debugger=None, backup:dict=None):
        # Setting
        backupstr = "w/o backup"
        self.changed = True
        if backup is not None:
            backupstr = "w/ backup"
            nout = backup['params']['nout']
            howmanystep = backup['params']['howmanystep']
            ncut = backup['params']['ncut']
            gal = backup['gal']          
            self.changed = False
        self.nout = nout # Descending order
        self.cat = copy.deepcopy(gal)
        self.id = self.cat['id']
        self.iout = self.cat['timestep']
        self._name = f"L{self.id} at {self.iout}"
        self.istep = len(nout) - np.where(nout == self.iout)[0][0]
        self.howmanystep = howmanystep
        self.ncut = ncut
        self.debugger = debugger
        self.verbose = verbose
        # func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}({self._name} {backupstr}) "
        # clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        # Particle info
        if backup is not None:
            self.nparts = backup['part']['nparts']
            self.pid = backup['part']['id']
            self.pvx = backup['part']['vx']
            self.pvy = backup['part']['vy']
            self.pvz = backup['part']['vz']
            self.pweight = backup['part']['weight']
        else:
            self.nparts = len(parts['id'])
            self.pid = np.abs(parts['id'])
            px, py, pz = parts['x'], parts['y'], parts['z']
            self.pvx, self.pvy, self.pvz = parts['vx', 'km/s'], parts['vy', 'km/s'], parts['vz', 'km/s']
            pm = parts['m']
            
            cx, cy, cz = self.cat['x'], self.cat['y'], self.cat['z']
            dist = distance3d(cx,cy,cz, px, py, pz) / self.cat['rvir']
            cvx, cvy, cvz = self.cat['vx'], self.cat['vy'], self.cat['vz']
            vels = distance3d(cvx,cvy,cvz, self.pvx, self.pvy, self.pvz)
            vels /= np.std(vels)
            dist = np.sqrt( dist**2 + vels**2 )
            self.pweight = pm / dist
            self.pweight /= np.sum(self.pweight)

        # Progenitor & Descendant
        if backup is not None:
            self.prog = self.cat['prog']
            self.prog_score = self.cat['prog_score']
            self.desc = self.cat['desc']
            self.desc_score = self.cat['desc_score']
            self.saved_matchrate = backup['saved']['matchrate']
            self.saved_veloffset = backup['saved']['veloffset']
        else:
            self.prog = None
            self.desc = None
            self.prog_score = None
            self.desc_score = None
            # for i in range(howmanystep):
            #     jstep = self.istep-i-1
            #     if jstep > 0:
            #         jout = nout[ len(nout)-jstep ]
            #         self.prog[jout] = None
            #         self.prog_score[jout] = None
            #     jstep = self.istep+i+1
            #     if jstep <= len(nout):
            #         jout = nout[ len(nout)-jstep ]
            #         self.desc[jout] = None
            #         self.desc_score[jout] = None
            self.saved_matchrate = {}
            self.saved_veloffset = {}
        # clock.done()
    
    def __del__(self):
        dprint_(f"[DEL] {self._name} is deleted", self.debugger)
            
    def name(self):
        return self._name

    def clear(self):
        self.nparts = None
        self.pid = None
        self.pvx = None
        self.pvy = None
        self.pvz = None
        self.pweight = None
        self.prog = None
        self.desc = None
        self.prog_score = None
        self.desc_score = None
        self.saved_matchrate = {}
        self.saved_veloffset = {}    


            
    def selfsave(self) -> dict:
        backup_keys = ["nparts", "id", "timestep", "aexp", "m", "x", "y", "z", "vx", "vy", "vz", "r", "rvir", "mvir"]
        lis = [ia for ia,ib in zip(self.cat, self.cat.dtype.names) if ib in backup_keys] + [self.prog] + [self.prog_score] + [self.desc] + [self.desc_score]
        dtype = np.dtype([idtype for idtype in self.cat.dtype.descr if idtype[0] in backup_keys] + [('prog','O'), ("prog_score",  'O'), ('desc','O'), ("desc_score",  'O')])
        arr = np.empty(1, dtype=dtype)[0]
        for i, ilis in enumerate(lis):
            arr[i] =  ilis
        # self.cat = np.record( np.rec.array(tuple(lis), dtype=dtype) )
        self.cat = arr

        backup = {}
        backup['gal'] = self.cat
        backup['params'] = {'nout':self.nout, 'ncut':self.ncut, 'howmanystep':self.howmanystep}
        backup['part'] = {'nparts':self.nparts, 'id':self.pid, 'vx':self.pvx, 'vy':self.pvy, 'vz':self.pvz, 'weight':self.pweight}
        backup['saved'] = {'matchrate':self.saved_matchrate, 'veloffset':self.saved_veloffset}

        return backup
    
    def calc_score(self, jout:int, otherleaves:list[Leaf], prefix=""):
        # for otherleaf in otherleaves:
        leng = len(otherleaves)
        ids = np.empty((leng,2), dtype=int)
        scores = np.empty((leng,5), dtype=float)
        for i in range(leng):
            otherleaf = otherleaves[i]
            if otherleaf.iout != jout:
                raise ValueError(f"Calc score at {jout} but found <{otherleaf.name}>!")
            score1, selfind = self.calc_matchrate(otherleaf, prefix="")
            score2, otherind = otherleaf.calc_matchrate(self, prefix="")
            score3 = self.calc_veloffset(otherleaf, selfind=selfind, otherind=otherind, prefix="")
            score4 = np.exp( -np.abs(np.log10(self.cat['m']/otherleaf.cat['m'])) ) 
            scores_tot = score1 + score2 + score3 + score4
            scores[i] = (scores_tot, score1, score2, score3, score4)
            ids[i] = (jout, otherleaf.id)
        arg = np.argsort(scores[:, 0])
        return ids[arg][::-1], scores[arg][::-1]

    def calc_matchrate(self, otherleaf:Leaf, prefix="") -> float:
        calc = True
        jout = otherleaf.iout
        if jout in self.saved_matchrate.keys():
            if otherleaf.id in self.saved_matchrate[jout].keys():
                val, ind = self.saved_matchrate[jout][otherleaf.id]
                calc = False
        if calc:        
            ind = large_isin(self.pid, otherleaf.pid)
            if not True in ind:
                val = -1
            else:
                val = np.sum( self.pweight[ind] )
            
            if not jout in self.saved_matchrate.keys():
                self.saved_matchrate[jout] = {}
                self.changed = True
            if not otherleaf.id in self.saved_matchrate[jout].keys():
                self.saved_matchrate[jout][otherleaf.id] = (val, ind)
                self.changed = True
        return val, ind

    def calc_bulkmotion(self, checkind=None, prefix=""):
        if checkind is None:
            checkind = np.full(self.nparts, True)

        weights = self.pweight[checkind]
        weights /= np.sum(weights)
        vx = np.convolve( self.pvx[checkind], weights[::-1], mode='valid' )[0] - self.cat['vx']
        vy = np.convolve( self.pvy[checkind], weights[::-1], mode='valid' )[0] - self.cat['vy']
        vz = np.convolve( self.pvz[checkind], weights[::-1], mode='valid' )[0] - self.cat['vz']

        return np.array([vx, vy, vz])
    
    def calc_veloffset(self, otherleaf:Leaf, selfind=None, otherind=None, prefix="") -> float:
        calc=True
        jout = otherleaf.iout
        if jout in self.saved_veloffset.keys():
            if otherleaf.id in self.saved_veloffset[jout].keys():
                val = self.saved_veloffset[jout][otherleaf.id]
                calc = False
        if calc:
            if selfind is None:
                val, selfind = self.calc_matchrate(otherleaf, prefix=prefix)
            if otherind is None:
                val, otherind = otherleaf.calc_matchrate(self, prefix=prefix)
            
            if howmany(selfind, True) < 3:
                val = 0
            else:
                selfv = self.calc_bulkmotion(checkind=selfind)
                otherv = otherleaf.calc_bulkmotion(checkind=otherind)
                val = 1 - nbnorm(otherv - selfv)/(nbnorm(selfv)+nbnorm(otherv))

            if not jout in self.saved_veloffset.keys():
                self.saved_veloffset[jout] = {}
                self.changed = True
            if not otherleaf.id in self.saved_veloffset[jout].keys():
                self.saved_veloffset[jout][otherleaf.id] = val
                self.changed = True
            if not self.iout in otherleaf.saved_veloffset.keys():
                otherleaf.saved_veloffset[self.iout] = {}
                otherleaf.changed = True
            if not otherleaf.id in otherleaf.saved_veloffset[self.iout].keys():
                otherleaf.saved_veloffset[self.iout][self.id] = val
                otherleaf.changed = True
        return val

        

