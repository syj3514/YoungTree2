import numpy as np
# from YoungTree3 import yroot
# from YoungTree3 import yleaf
from YoungTree3.ytool import *
# from ytool import *
from numpy.lib.recfunctions import append_fields
from tqdm import tqdm
import matplotlib.pyplot as plt
from rur import uri, uhmi

params = importlib.import_module("params_default")
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


yess = ["Y","y","yes", "YES", "Yes"]
galdict = {}
yrepo = f"{repo}/YoungTree"

ref = time.time()

debugger = None
fname = make_logname(mode, -1, logprefix=p.logprefix, specific="compile", dirname=yrepo)
debugger = custom_debugger(fname, detail=p.detail)
inihandlers = debugger.handlers
message = f"< YoungTree >\nUsing {modename} {galstr}\nCompile start\n\nSee `{fname}`\n\n"
debugger.info(message)
print(message)














#########################################################
#   Connect each leaf
#########################################################
#   1. Start from last output(=aout)
#   2. If one agal doesn't have any progenitors (prog=0), set `last`=0
#       2-1. If at last output, set `last`=int( {iout:05d}{id:05d} ) (ex: 0018700001)
#   3. If one agal's `last`>0, check these conditions
#       i) at least one 1step progenitor (at bout) -> Find agal's max-score-progenitor at bout (bgal)
#       ii) bgal's max-score-descendant == agal
#       iii) at least one 2step progenitor (at cout) -> Find agal's max-score-progenitor at cout (cgal)
#       iv) bgal's max-score-progenitor == cgal
#       v) cgal's max-score-descendant == bgal
#   4. If all above conditions are satisfied, set `last` of a,b,cgals as a's `last`
#   5. Iterate 3 to 4
# aout = 187
def maxprog(gal, iout, galdict=None):
    arg = gal['prog'][:,0] == iout # True?
    prog = gal['prog'][arg]
    if len(prog)>0:
        prog_score = gal['prog_score'][arg]
        argmax = np.argmax(prog_score)
        bid = prog[argmax][1]
        if bid >0:
            return galdict[iout][bid-1]
def maxdesc(gal, iout, galdict=None):
    arg = gal['desc'][:,0] == iout # True?
    desc = gal['desc'][arg]
    if len(desc)>0:
        desc_score = gal['desc_score'][arg]
        argmax = np.argmax(desc_score)
        bid = desc[argmax][1]
        if bid >0:
            return galdict[iout][bid-1]
        
go=True
if os.path.isfile(f"{yrepo}/ytree_last_strict.pickle"):
    ans=input(f"You already have `{yrepo}/ytree_last_strict.pickle`. Ovewrite? [Y/N]")
    go = ans in yess
if go:
    dprint_("Connect most stable leaves...", debugger, level='info')
    #############################################
    # Ex:
    #       aout = 187
    #       bout = 186
    #       cout = 185
    #       dout = 184
    #       eout = 183
    #       fout = 182
    #############################################
    frac = [0, 0, 0]
    iterobj = tqdm(nout[:-1], f"{frac}",)
    for aout in iterobj:
        dprint_(f"[{aout}]", debugger, level='info')
        fname = f"{yrepo}/ytree_{aout:05d}.pickle"
        if not aout in galdict.keys():
            galdict[aout] = pklload(fname)
            temp = np.zeros(len(galdict[aout]), dtype=int) - 1
            galdict[aout] = append_fields(galdict[aout], "last", temp, usemask=False)
        astep = out2step(aout, galaxy=True, mode=mode)
        agals = galdict[aout]
        for i in range(5):
            bstep = astep-1-i
            if bstep >= np.min(nstep):
                bout = step2out(bstep, galaxy=True, mode=mode)
                fname = f"{yrepo}/ytree_{bout:05d}.pickle"
                if not bout in galdict.keys():
                    galdict[bout] = pklload(fname)
                    temp = np.zeros(len(galdict[bout]), dtype=int) - 1
                    galdict[bout] = append_fields(galdict[bout], "last", temp, usemask=False)

        iterobj.desc = f"Total {frac[0]:04d}, {frac[1]:04d}, {frac[2]:04d}"
        frac = [0, 0, 0]
        frac[0] = len(agals)
        for agal in agals:
            if np.sum(agal['prog'][:,1]) == 0:
                if agal['last'] < 0:
                    dprint_(f"[{agal['id']:05d}]\tNo progenitor (-1 --> 0)", debugger, level='debug')
                    agal['last'] = 0
                else:
                    dprint_(f"[{agal['id']:05d}]\tNo progenitor ({agal['last']})", debugger, level='debug')
            try:
                assert agal['last'] != 0
                aid = agal['id']
                last = agal['last']
                lastid = gal2id(agal) if last<0 else last
                bstep = astep-1
                bout = step2out(bstep, galaxy=True, mode=mode)
                bgal = maxprog(agal, bout, galdict=galdict)
                assert bgal is not None
                bid = bgal['id']
                blast = bgal['last']
                progs = agal['prog'][agal['prog'][:,0] == bout]
                descs = bgal['desc'][bgal['desc'][:,0] == aout]
                ##################################################
                #       Only one case
                ##################################################
                #   Agal has only one prog B at bout
                #   Bgal has only one desc A at aout
                if(progs.shape[0]==1)&(descs.shape[0]==1)&(progs[0][1]==bgal['id'])&(descs[0][1]==agal['id']):
                    frac[1] += 1
                    galdict[bout][bid-1]['last'] = lastid
                    galdict[aout][aid-1]['last'] = lastid
                    if last==lastid:
                        txt = f"({last})"
                    else:
                        txt = f"({last} --> {lastid})"
                    dprint_(f"[{agal['id']:05d}]\tTight one connection to {bid} at {bout} {txt}\t[{bid} at {bout}] ({blast} --> {lastid})", debugger, level='debug')
                ##################################################
                #       Else
                ##################################################
                #   1. Find Agal's progenitor: AB, AC, AD, AE, AF
                #       i) Their desc is Agal
                #           > AB's desc at aout == A
                #           > AC's desc at aout == A
                #           > AD's desc at aout == A
                #           > AE's desc at aout == A
                #           > AF's desc at aout == A
                #   2. Find Bgal's progenitor: BC, BD, BE, BF
                #       ii) Agal's progenitors are Bgal's progenitors
                #           > BC == AC
                #           > BD == AD
                #           > BE == AE
                #           > BF == AF
                #           > BC == AC
                #       iii) Their desc is B
                #           > AC's desc at bout == B
                #           > AD's desc at bout == B
                #           > AE's desc at bout == B
                #           > AF's desc at bout == B
                #   3. If all conditions are satisfied, connect A<->B
                else:
                    ################################
                    # A <-> A's progs Connection
                    ################################
                    # Find A's 5 progs
                    nprog = min(5, astep-np.min(nstep))
                    progs = [None]*nprog
                    for i in range(nprog):
                        bstep = astep-1-i
                        bout = step2out(bstep, galaxy=True, mode=mode)
                        bgal = maxprog(agal, bout, galdict=galdict)
                        assert bgal is not None
                        progs[i] = bgal
                    # They think their desc is A
                    for prog in progs:
                        check = maxdesc(prog, aout, galdict=galdict)
                        assert check is not None
                        assert check['id'] == agal['id']
                    ################################
                    # B <-> A's progs Connection    
                    ################################
                    # At B step
                    bstep = astep-1
                    bout = step2out(bstep, galaxy=True, mode=mode)
                    for i in range(nprog-1):
                        # At B step, bgal's prog at C
                        # C = (A-2) <-> (A-1)'s 1th prog
                        # C = (A-3) <-> (A-1)'s 2th prog
                        # C = (A-4) <-> (A-1)'s 3th prog
                        # C = (A-5) <-> (A-1)'s 4th prog
                        cstep = astep-2-i
                        cout = step2out(cstep, galaxy=True, mode=mode)
                        cgal = maxprog(progs[0], cout, galdict=galdict)
                        assert cgal is not None
                        assert cgal['id'] == progs[i+1]['id']
                        # At C step, cgal's desc at B
                        # B = (A-1) <-> (A-2)'s 1th desc
                        # B = (A-1) <-> (A-3)'s 2th desc
                        # B = (A-1) <-> (A-4)'s 3th desc
                        # B = (A-1) <-> (A-5)'s 4th desc
                        bgal = maxdesc(progs[i+1], bout, galdict=galdict)
                        assert bgal is not None
                        assert bgal['id'] == progs[0]['id']
                    bout = progs[0]['timestep']
                    bid = progs[0]['id']
                    blast = progs[0]['last']
                    if galdict[bout][bid-1]['last']>0:
                        print(f"<{bid} at {bout}> changes its branch {galdict[bout][bid-1]['last']}->{lastid}")
                        dprint_(f"<{bid} at {bout}> changes its branch {galdict[bout][bid-1]['last']}->{lastid}", debugger, level='warning')
                    frac[2] += 1
                    galdict[bout][bid-1]['last'] = lastid
                    galdict[aout][aid-1]['last'] = lastid
                    if last==lastid:
                        txt = f"({last})"
                    else:
                        txt = f"({last} --> {lastid})"
                    dprint_(f"[{agal['id']:05d}]\tTight 5 connections to {bid} at {bout} {txt}\t[{bid} at {bout}] ({blast} --> {lastid})", debugger, level='debug')

                    success = True
            except AssertionError as e:
                dprint_(f"[{agal['id']:05d}]\tFail to find stable progenitor ({last})", debugger, level='debug')
            except Exception as e:
                traceback.print_exception(e)
                # pass
        pklsave(galdict[aout], f"{yrepo}/ytree_{aout:05d}_last_strict.pickle", overwrite=True)
        dprint_(f"`{yrepo}/ytree_{aout:05d}_last_strict.pickle` saved\n", debugger, level='info')
        del galdict[aout]
    keys = list(galdict.keys())
    if len(keys)>0:
        dprint_(f"Save remained iout: {keys}", debugger, level='info')
        for key in keys:
            pklsave(galdict[key], f"{yrepo}/ytree_{key:05d}_last_strict.pickle", overwrite=True)
            dprint_(f"`{yrepo}/ytree_{key:05d}_last_strict.pickle` saved\n", debugger, level='info')
            del galdict[key]

    dprint_("\nMerge all treebricks...", debugger, level='info')
    for iout in tqdm(nout):
        igals = pklload(f"{yrepo}/ytree_{iout:05d}_last_strict.pickle")
        gals = igals if iout==np.max(nout) else np.hstack((gals, igals))
        os.remove(f"{yrepo}/ytree_{iout:05d}_last_strict.pickle")
        dprint_(f"`{yrepo}/ytree_{iout:05d}_last_strict.pickle` deleted", debugger, level='info')
    pklsave(gals, f"{yrepo}/ytree_last_strict.pickle", overwrite=True)
    dprint_(f"`{yrepo}/ytree_last_strict.pickle` saved\n", debugger, level='info')

gals = pklload(f"{yrepo}/ytree_last_strict.pickle")
    










def _prog(gal, iout, all=False):
    progs = gal['prog'][gal['prog'][:,0] == iout]
    if len(progs)==0:
        return None
    if all:
        return progs
    prog_scores = gal['prog_score'][gal['prog'][:,0] == iout]
    argmax = np.argmax(prog_scores)
    return progs[argmax]

def _desc(gal, iout, all=False):
    descs = gal['desc'][gal['desc'][:,0] == iout]
    if len(descs)==0:
        return None
    if all:
        return descs
    desc_scores = gal['desc_score'][gal['desc'][:,0] == iout]
    argmax = np.argmax(desc_scores)
    return descs[argmax]

def _gal(iout, gid, ytree):
    return ytree[ytree['timestep']==iout][gid-1]

def _last(iout, gid, ytree):
    return _gal(iout, gid, ytree)['last']

def connected(agal, bgal, ytree):
    if len(agal)==2:
        agal = _gal(agal[0], agal[1], ytree)
    aout, aid = agal['timestep'], agal['id']
    if len(bgal)==2:
        bgal = _gal(bgal[0], bgal[1], ytree)
    bout, bid = bgal['timestep'], bgal['id']
    
    result = False
    if aout>bout:
        aprog = _prog(agal, bout)
        bdesc = _desc(bgal, aout)
        if (aprog is None) or (bdesc is None):
            return False
        result = (aprog[1]==bid)&(bdesc[1]==aid)
    elif aout<bout:
        adesc = _desc(agal, bout)
        bprog = _prog(bgal, aout)
        if (bprog is None) or (adesc is None):
            return False
        result = (adesc[1]==bid)&(bprog[1]==aid)
    return result

def occupied(iout, last, ytree):
    inst = ytree[ytree['last'] == last]['timestep']
    return iout in inst


def direct_cand(ytree, lastid):
    global mode, nout, nstep
    pbranch, pscore, dbranch, dscore = -1,-1,-1,-1
    current = ytree[ytree['last'] == lastid]
    first = current[-1]
    ta = out2step(first['timestep'], galaxy=p.galaxy, mode=mode, nout=nout, nstep=nstep); tb=-1
    if first['prog'] is not None:
        branchs = np.array([_gal(prog[0], prog[1], ytree)['last'] for prog in first['prog']])
        unique = np.unique(branchs[branchs != None]); unique=unique[unique>0]
        if len(unique)>0:
            scores = np.zeros(len(unique))
            for i in range(len(unique)):
                scores[i] = np.sum( first['prog_score'][branchs == unique[i]][:,0] )
            argmax = np.argmax(scores)
            pbranch, pscore = unique[argmax], scores[argmax]

            current = ytree[ytree['last'] == pbranch]
            final = current[0] ####################### This should be connect to next branch??????????????
            tb = out2step(final['timestep'], galaxy=p.galaxy, mode=mode, nout=nout, nstep=nstep)
            if final['desc'] is not None:
                branchs = np.array([_gal(desc[0], desc[1], ytree)['last'] for desc in final['desc']])
                unique = np.unique(branchs[branchs != None]); unique=unique[unique>0]
                if len(unique)>0:
                    scores = np.zeros(len(unique))
                    for i in range(len(unique)):
                        scores[i] = np.sum( final['desc_score'][branchs == unique[i]][:,0] )
                    argmax = np.argmax(scores)
                    dbranch, dscore = unique[argmax], scores[argmax]
    return pbranch, pscore, dbranch, dscore, ta, tb












go=True
if os.path.isfile(f"{yrepo}/ytree_last_connect.pickle"):
    ans=input(f"You already have `{yrepo}/ytree_last_connect.pickle`. Ovewrite? [Y/N]")
    go = ans in yess
if go:
    removed = []
    lasts = gals['last'][(gals['last']>0)]
    lasts = np.unique(lasts)
    dprint_(f"{len(lasts)} branches are built --> connect these branches", debugger, level='info')
    for last in tqdm(lasts[::-1]):
        if not last in removed:
            pbranch, pscore, dbranch, dscore, ta, tb = direct_cand(gals, last)
            if last == dbranch:
                if (last>0)&(dbranch>0):
                    if ta-tb == 1: # Shut up and just connect
                        temp = gals['last']
                        temp[temp == pbranch] = last
                        gals['last'] = temp
                        removed.append(pbranch)
                        dprint_(f"({pbranch}) --> ({last}) with {ta-tb} offset", debugger, level='debug')

                    elif ta>tb:
                        ibranch = gals[gals['last'] == last]
                        igal = ibranch[-1]
                        istep = out2step(igal['timestep'], galaxy=True, mode=mode, nout=nout, nstep=nstep)
                        jbranch = gals[gals['last'] == pbranch]
                        jgal = jbranch[0]
                        jstep = out2step(jgal['timestep'], galaxy=True, mode=mode, nout=nout, nstep=nstep)
                        
                        btw = istep-jstep-1
                        btws = [0]*btw
                        result = True
                        for ith in range(btw):
                            kstep = istep - ith - 1
                            kout = step2out(kstep, galaxy=True, mode=mode, nout=nout, nstep=nstep)
                            kj = _desc(jgal, kout); ki = _prog(igal, kout)
                            kgal = _gal(ki[0], ki[1], gals)
                            result *= np.array_equal(ki,kj)&(connected(igal, ki, gals))&(connected(jgal, kj, gals))&(kgal['last']<1)
                            btws[ith] = kj
                        if result:
                            temp = gals['last']
                            temp[temp == pbranch] = last
                            for ith in range(btw):
                                where = np.where( (gals['timestep']==btws[ith][0]) & (gals['id']==btws[ith][1]) )[0][0]
                                temp[where] = last
                            gals['last'] = temp
                            removed.append(pbranch)
                            dprint_(f"({pbranch}) --> ({last}) with {ta-tb} offset", debugger, level='debug')

                    elif ta==tb: # Reject for one overlapped snapshot
                        pass
                    else: # Reject for many overlapped snapshots
                        pass
    lasts = gals['last'][(gals['last']>0)]
    lasts = np.unique(lasts)
    dprint_(f"{len(lasts)} branches are remained", debugger, level='info')
    pklsave(gals, f"{yrepo}/ytree_last_connect.pickle", overwrite=True)
    dprint_(f"`{yrepo}/ytree_last_connect.pickle` saved\n", debugger, level='info')
gals = pklload(f"{yrepo}/ytree_last_connect.pickle")


    








go=True
if os.path.isfile(f"{yrepo}/ytree_remove_zero.pickle"):
    ans=input(f"You already have `{yrepo}/ytree_remove_zero.pickle`. Ovewrite? [Y/N]")
    go = ans in yess
if go:
    dprint_(f"\nRemove zero-branch", debugger, level='info')
    gals = append_fields(gals, "merged", np.zeros(len(gals), dtype=int),usemask=False)
    zeroind = np.where(gals['last']==0)[0]
    for ind in tqdm(zeroind):
        iout = gals[ind]['timestep']; iid = gals[ind]['id']
        descs = gals[ind]['desc']
        if descs is not None:
            ############################################
            # No descendants
            ############################################
            if all(descs[:, 1] == 0):
                # All descs are 0
                gals[ind]['last'] = gal2id(gals[ind])
                dprint_(f"[{gals[ind]['id']} at {gals[ind]['timestep']}] 0 --newlast-> ({gals[ind]['last']}) (No descendants)", debugger, level='debug')

            ############################################
            # Yes descendants
            ############################################
            else:
                arg = descs[:,1] != 0
                descs = descs[arg] # nonzero descs
                dlasts = np.array([_last(d[0], d[1], gals) for d in descs])
                ###################
                # Only one branch
                ###################
                if len(np.unique(dlasts)) == 1:
                    dlast = dlasts[0]
                    if dlast>0:
                        # Normal branch
                        if occupied(iout, dlast, gals):
                            gals[ind]['merged'] = gal2id(gals[ind])#int( descs[0][0]*100000 + descs[0][1] )
                            dprint_(f"[{gals[ind]['id']} at {gals[ind]['timestep']}] 0 --merged--> ({dlast}) (Same descendants)", debugger, level='debug')
                        else:
                            dprint_(f"[{gals[ind]['id']} at {gals[ind]['timestep']}] 0 --connect-> ({dlast}) (Same descendants)", debugger, level='debug')
                        gals[ind]['last'] = dlast

                    else:
                        # All branchs = -1
                        leng = descs.shape[0]
                        temp = [gal2id(gals[ind])] + [int( descs[i][0]*100000 + descs[i][1] ) for i in range(leng)]
                        for i in range(leng):
                            if connected(gals[ind], descs[i], gals):
                                temp[0] = int( descs[i][0]*100000 + descs[i][1] )
                            for j in range(leng-i-1):
                                if connected(descs[i], descs[i+j+1], gals):
                                    temp[i+1] = int( descs[i+j+1][0]*100000 + descs[i+j+1][1] )
                        gals[ind]['last'] = temp[0]
                        dprint_(f"[{gals[ind]['id']} at {gals[ind]['timestep']}] 0 --newlast-> ({temp[0]}) (Orphan descendants)", debugger, level='debug')
                        for i in range(leng):
                            desc = descs[i]
                            where = np.where((gals['timestep']==desc[0])&(gals['id']==desc[1]))[0][0]
                            gals[where]['last'] = temp[i+1]
                            dprint_(f"[{gals[where]['id']} at {gals[where]['timestep']}] -1 --newlast-> ({temp[i+1]}) (Orphan descendants)", debugger, level='debug')
                        
                ###################
                # More than 1 branch
                ###################
                else:
                    leng = descs.shape[0]
                    temp = [int( descs[i][0]*100000 + descs[i][1] ) if _gal(descs[i][0],descs[i][1],gals)['last']<=0 else _gal(descs[i][0],descs[i][1],gals)['last'] for i in range(leng)]
                    temp = [gal2id(gals[ind])] + temp
                    for i in range(leng):
                        if connected(gals[ind], descs[i], gals):
                            temp[0] = temp[i+1]
                        for j in range(leng-i-1):
                            if connected(descs[i], descs[i+j+1], gals):
                                temp[i+1] = temp[i+j+1]
                    
                    descs = np.vstack((np.array([[iout, iid]]), descs[dlasts<=0]))
                    temp2 = np.array(temp[1:])
                    temp = [temp[0]] + (temp2[dlasts<=0]).tolist()
                    unique, ucount = np.unique(temp, return_counts=True)
                    for uni, uco in zip(unique, ucount):
                        ddescs = descs[temp == uni]
                        final = ddescs[np.argmax(ddescs[:,0])]
                        sole = True
                        for i in range(uco):
                            sole = sole or ~occupied(ddescs[i][0], uni, gals)
                        for i in range(uco):
                            desc = ddescs[i]
                            where = np.where((gals['timestep']==desc[0])&(gals['id']==desc[1]))[0][0]
                            if sole:
                                dprint_(f"[{gals[where]['id']} at {gals[where]['timestep']}] {gals[where]['last']} --newlast-> ({uni}) (Complex case)", debugger, level='debug')
                            else:
                                gals[where]['merged'] = int( final[0]*100000 + final[1] )
                                dprint_(f"[{gals[where]['id']} at {gals[where]['timestep']}] {gals[where]['last']} --merged--> ({uni}) (Complex case)", debugger, level='debug')
                            gals[where]['last'] = uni
                            
        else:
            # Last Snapshot
            gals[ind]['last'] = gal2id(gals[ind])
            dprint_(f"[{gals[ind]['id']} at {gals[ind]['timestep']}] 0 --newlast-> ({gals[ind]['last']}) (Last snapshot)", debugger, level='debug')
    pklsave(gals, f"{yrepo}/ytree_remove_zero.pickle", overwrite=True)
    dprint_(f"`{yrepo}/ytree_remove_zero.pickle` saved\n", debugger, level='info')       
gals = pklload(f"{yrepo}/ytree_remove_zero.pickle")







go=True
if os.path.isfile(f"{yrepo}/ytree_remove_negative.pickle"):
    ans=input(f"You already have `{yrepo}/ytree_remove_negative.pickle`. Ovewrite? [Y/N]")
    go = ans in yess
if go:
    dprint_(f"\nRemove negative-branch", debugger, level='info')

    indices = np.where(gals['last'] == -1)[0]
    orphans = gals[indices]
    iouts = np.unique(orphans['timestep'])[::-1]
    for iout in tqdm(iouts):
        dprint_(f"[{iout:05d}]", debugger, level='info')
        orps = orphans[orphans['timestep'] == iout]
        inds = indices[orphans['timestep'] == iout]
        for orp, ind in zip(orps, inds):
            success = False
            if orp['last']<0:
                gals[ind]['last'] = gal2id(gals[ind])
            descs = orp['desc']
            progs = orp['prog']
            if (descs is not None) and (not all(descs[:,1] <= 0)):
                desc_scores = orp['desc_score'][:,0]
                desc_scores = desc_scores[descs[:,1] > 0]
                descs = descs[descs[:,1] > 0]
                dgals = [_gal(desc[0], desc[1], gals) for desc in descs]
                cands = np.zeros(len(dgals), dtype=int)
                for i, dgal in enumerate(dgals):
                    dout = dgal['timestep']; did = dgal['id']; dlast = dgal['last']; dmerge = dgal['merged']
                    if connected(orp, dgal, gals):
                        cands[i] = dlast
                if not np.array_equal(np.zeros(len(dgals), dtype=int), cands):
                    ucands = np.unique(cands[cands != 0])
                    leng = len(ucands)
                    if leng == 1:
                        decision = ucands[0]
                    else:
                        scores = np.zeros(leng)
                        for i in range(leng):
                            cand = ucands[i]
                            assert cand > 0
                            inst = gals[gals['last'] == cand]
                            maxout = np.max(descs[:,0]); minout = np.min(progs[:,0])
                            inst = inst[ (inst['timestep'] <= maxout) & (inst['timestep'] >= minout) ]
                            for ins in inst:
                                check = (descs[:,0]==ins['timestep']) & (descs[:,1]==ins['id'])
                                if True in check:
                                    scores[i] += desc_scores[check][0]
                        decision = ucands[np.argmax(scores)]
                    where = np.where(cands==decision)[0]
                    if occupied(iout, decision, gals):
                        most = descs[np.argmax(desc_scores)]
                        merged = _gal(most[0], most[1], gals)['merged']
                        gals[ind]['merged'] = gal2id(gals[ind]) if merged==0 else merged
                        dprint_(f"\t[{orp['id']:05d}] --merged--> ({gals[ind]['last']})", debugger, level='debug')
                    else:
                        dprint_(f"\t[{orp['id']:05d}] --connect-> ({gals[ind]['last']})", debugger, level='debug')
                    gals[ind]['last'] = decision
                    success = True
            if not success:
                dprint_(f"\t[{orp['id']:05d}] Fail -> lonely branch --> ({gals[ind]['last']})", debugger, level='debug')
    pklsave(gals, f"{yrepo}/ytree_remove_negative.pickle", overwrite=True)
    dprint_(f"`{yrepo}/ytree_remove_negative.pickle` saved\n", debugger, level='info')       
gals = pklload(f"{yrepo}/ytree_remove_negative.pickle")


print(f"\nDone ({time.time()-ref:.3f} sec)\n")
dprint_(f"\nDone ({time.time()-ref:.3f} sec)\n", debugger, level='info')