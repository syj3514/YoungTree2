import numpy as np
# from YoungTree3 import yroot
# from YoungTree3 import yleaf
from YoungTree3.ytool import *
# from ytool import *
from numpy.lib.recfunctions import append_fields
from tqdm import tqdm
import matplotlib.pyplot as plt
from rur import uri, uhmi

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

#########################################################
#   Mode Configuration
#########################################################
mode = sys.argv[1]
if len(sys.argv) < 2:
    mode = input("mode=? ")
if not mode in modenames.keys():
    raise ValueError(f"`{mode}` is not supported!\nSee {list(modenames.keys())}")
modename = modenames[mode]
repo, rurmode, dp = mode2repo(mode)
# For message printing
galstr = "Halo"
galstrs = "Halos"
if p.galaxy:
    galstr = "Galaxy"
    galstrs = "Galaxies"

# Read output list
nout = load_nout(mode=mode, galaxy=p.galaxy)
nstep = load_nstep(mode=mode, galaxy=p.galaxy, nout=nout)



def gal2id(gal):
    # return f"{gal['timestep']:05d}{gal['id']:05d}"
    return gal['timestep']*100000 + gal['id']
def id2gal(iid):
    return iid//100000, iid%100000



galdict = {}
yrepo = f"{repo}/YoungTree"
if os.path.isfile(f"{yrepo}/ytree_last_strict.pickle"):
    sys.exit(f"You already have `{yrepo}/ytree_last_strict.pickle`")


# aout = 187
for aout in tqdm(nout[:-1]):
    fname = f"{yrepo}/ytree_{aout:05d}.pickle"
    if not aout in galdict.keys():
        galdict[aout] = pklload(fname)
        inst = np.zeros(len(galdict[aout]), dtype=int) - 1
        galdict[aout] = append_fields(galdict[aout], "last", inst, usemask=False)
        # galdict[aout] = append_fields(galdict[aout], "last_score", inst, usemask=False)
    astep = out2step(aout, galaxy=True, mode=mode)
    agals = galdict[aout]

    bstep = astep-1
    if bstep >= np.min(nstep):
        bout = step2out(bstep, galaxy=True, mode=mode)
        fname = f"{yrepo}/ytree_{bout:05d}.pickle"
        if not bout in galdict.keys():
            galdict[bout] = pklload(fname)
            inst = np.zeros(len(galdict[bout]), dtype=int) - 1
            galdict[bout] = append_fields(galdict[bout], "last", inst, usemask=False)
    cstep = astep-2
    if cstep >= np.min(nstep):
        cout = step2out(cstep, galaxy=True, mode=mode)
        fname = f"{yrepo}/ytree_{cout:05d}.pickle"
        if not cout in galdict.keys():
            galdict[cout] = pklload(fname)
            inst = np.zeros(len(galdict[cout]), dtype=int) - 1
            galdict[cout] = append_fields(galdict[cout], "last", inst, usemask=False)
    
    for agal in agals:
        if np.sum(agal['prog'][:,1]) == 0:
            agal['last'] = 0
        try:
            assert agal['last'] != 0
            aid = agal['id']
            last = agal['last']
            lastid = gal2id(agal) if last<0 else last
            # Find B gal
            a2b_arg = agal['prog'][:,0] == bout # True?
            a2b_prog = agal['prog'][a2b_arg]
            assert len(a2b_prog)>0
            a2b_prog_score = agal['prog_score'][a2b_arg]
            a2b_argmax = np.argmax(a2b_prog_score)
            bid = a2b_prog[a2b_argmax][1]
            assert bid > 0
            bgal = galdict[bout][bid-1]
            # B's desc = A?
            b2a_arg = bgal['desc'][:,0] == aout
            b2a_desc = bgal['desc'][b2a_arg]
            b2a_desc_score = bgal['desc_score'][b2a_arg]
            b2a_argmax = np.argmax(b2a_desc_score)
            aid = b2a_desc[b2a_argmax][1]
            assert aid == agal['id']
            # Find C gal
            a2c_arg = agal['prog'][:,0] == cout # True?
            a2c_prog = agal['prog'][a2c_arg]
            a2c_prog_score = agal['prog_score'][a2c_arg]
            a2c_argmax = np.argmax(a2c_prog_score)
            cid = a2c_prog[a2c_argmax][1]
            assert cid > 0
            cgal = galdict[cout][cid-1]
            # C's desc = B?
            c2b_arg = cgal['desc'][:,0] == bout # True?
            c2b_desc = cgal['desc'][c2b_arg]
            c2b_desc_score = cgal['desc_score'][c2b_arg]
            c2b_argmax = np.argmax(c2b_desc_score)
            assert c2b_desc[c2b_argmax][1] == bgal['id']
            # B's prog = C?
            b2c_arg = bgal['prog'][:,0] == cout # True?
            b2c_prog = bgal['prog'][b2c_arg]
            b2c_prog_score = bgal['prog_score'][b2c_arg]
            b2c_argmax = np.argmax(b2c_prog_score)
            assert b2c_prog[b2c_argmax][1] == cgal['id']
            if galdict[bout][bid-1]['last']>0:
                if galdict[bout][bid-1]['last'] != lastid:
                    print(f"{galdict[bout][bid-1]['last']} != {lastid}")
                    raise ValueError(f"{galdict[bout][bid-1]['last']} != {lastid}")
            galdict[bout][bid-1]['last'] = lastid
            galdict[aout][aid-1]['last'] = lastid
            galdict[cout][cid-1]['last'] = lastid
            
        except:
            pass
    pklsave(galdict[aout], f"{yrepo}/ytree_{aout:05d}_last_strict.pickle", overwrite=True)
    del galdict[aout]
keys = list(galdict.keys())
if len(keys)>0:
    for key in keys:
        pklsave(galdict[key], f"{yrepo}/ytree_{key:05d}_last_strict.pickle", overwrite=True)
        del galdict[key]

for iout in tqdm(nout):
    igals = pklload(f"{yrepo}/ytree_{iout:05d}_last_strict.pickle")
    gals = igals if iout==np.max(nout) else np.hstack((gals, igals))
    os.remove(f"{yrepo}/ytree_{iout:05d}_last_strict.pickle")
pklsave(gals, f"{yrepo}/ytree_last_strict.pickle")