import numpy as np
# from YoungTree3 import yroot
# from YoungTree3 import yleaf
from YoungTree3.ytool import *
# from ytool import *
from numpy.lib.recfunctions import append_fields
from tqdm import tqdm
import matplotlib.pyplot as plt
from rur import uri, uhmi

print("\n$ python3 ycompile.py <MODE> <PARAMS.py>\n")

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
yrepo = f"{repo}/YoungTree"
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
debugger = None
fname = make_logname(mode, -1, logprefix=p.logprefix, specific="compile", dirname=yrepo)
debugger = custom_debugger(fname, detail=p.detail)
inihandlers = debugger.handlers
message = f"< YoungTree >\nUsing {modename} {galstr}\nCompile start\n\nSee `{fname}`\n\n"
debugger.info(message)
print(message)




#########################################################
#   Compile all segments
#########################################################
yess = ["Y","y","yes", "YES", "Yes"]
ref = time.time()
go=True
if os.path.isfile(f"{yrepo}/ytree_all.pickle"):
    ans=input(f"You already have `{yrepo}/ytree_all.pickle`. Ovewrite? [Y/N]")
    go = ans in yess
if go:
    dprint_("Gather all files...", debugger, level='info')
    for i, iout in enumerate(nout):
        file = pklload(f"{yrepo}/ytree_{iout:05d}.pickle")
        if i==0:
            gals = file
        else:
            gals = np.hstack((gals, file))
    dprint_("Add column `last`...", debugger, level='info')
    temp = np.zeros(len(gals), dtype=np.int32)
    gals = append_fields(gals, "last", temp, usemask=False)
    dprint_("Convert `prog` and `desc` to easier format...", debugger, level='info')
    for gal in tqdm(gals):
        if gal['prog'] is None:
            gal['prog'] = np.array([], dtype=np.int32)
            gal['prog_score'] = np.array([], dtype=np.float64)
        else:
            arg = gal['prog'][:,1]>0
            prog = gal['prog'][:,0]*100000 + gal['prog'][:,1]
            progscore = gal['prog_score'][:,0]
            gal['prog'] = prog[arg].astype(np.int32)
            gal['prog_score'] = progscore[arg].astype(np.float64)
        
        if gal['desc'] is None:
            gal['desc'] = np.array([], dtype=np.int32)
            gal['desc_score'] = np.array([], dtype=np.float64)
        else:
            arg = gal['desc'][:,1]>0
            desc = gal['desc'][:,0]*100000 + gal['desc'][:,1]
            descscore = gal['desc_score'][:,0]
            gal['desc'] = desc[arg].astype(np.int32)
            gal['desc_score'] = descscore[arg].astype(np.float64)
    pklsave(gals, f"{yrepo}/ytree_all.pickle", overwrite=True)
    dprint_(f"`{yrepo}/ytree_all.pickle` saved\n", debugger, level='info')
gals = pklload(f"{yrepo}/ytree_all.pickle")






#########################################################
#   Connect branch
#########################################################
go=True
if os.path.isfile(f"{yrepo}/ytree_stable.pickle"):
    ans=input(f"You already have `{yrepo}/ytree_stable.pickle`. Ovewrite? [Y/N]")
    go = ans in yess
if go:
    dprint_("Make dictionary...", debugger, level='info')
    gals = append_fields(gals, "from", np.zeros(len(gals), dtype=np.int32), usemask=False)
    gals = append_fields(gals, "fat", np.zeros(len(gals), dtype=np.int32), usemask=False)
    gals = append_fields(gals, "son", np.zeros(len(gals), dtype=np.int32), usemask=False)
    gals = append_fields(gals, "fat_score", np.zeros(len(gals), dtype=np.float64), usemask=False)
    gals = append_fields(gals, "son_score", np.zeros(len(gals), dtype=np.float64), usemask=False)
    inst = {}
    for iout in tqdm(nout):
        inst[iout] = gals[gals['timestep']==iout]
    gals = None

    dprint_("Find son & father...", debugger, level='info')
    offsets = np.array([1,2,3,4,5])
    for offset in offsets:
        dprint_(f"\n{offset}\n", debugger)
        iterobj = tqdm(nout, desc=f"(offset {offset})")
        for iout in iterobj:
            temp = inst[iout]
            do = np.ones(len(temp), dtype=bool)
            for ihalo in temp:
                if ihalo['son']<=0:
                    iid = gal2id(ihalo)
                    do[ihalo['id']-1] = False
                    idesc, idscore = maxdesc(ihalo, all=False, offset=offset)
                    if idesc==0:
                        dprint_(f"\t{iid} No desc", debugger)
                        pass
                    else:
                        dhalo = gethalo(idesc//100000, idesc%100000, halos=inst)
                        prog, pscore = maxprog(dhalo, all=False, offset=offset)
                        if prog==iid: # each other
                            nrival = 0
                            for jhalo, ido in zip(temp, do):
                                if (idesc in jhalo['desc'])&(jhalo['son']<=0)&(ido):
                                    jid = gal2id(jhalo)
                                    if jid != iid:
                                        jdesc, jdscore = maxdesc(jhalo, all=False, offset=offset)
                                        if jdesc==idesc:
                                            do[jhalo['id']-1] = False
                                            nrival += 1
                                            if jhalo['son'] == 0:
                                                dprint_(f"\t\trival {jid} newly have son {-idesc} ({jdscore:.4f})", debugger)
                                                jhalo['son'] = -idesc
                                                jhalo['son_score'] = jdscore
                                            elif jhalo['son'] < 0:
                                                if jhalo['son_score'] > jdscore:
                                                    dprint_(f"\t\trival {jid} keep original son {jhalo['son']} ({jhalo['son_score']:.4f}) rather than {-idesc} ({jdscore:.4f})", debugger)
                                                else:
                                                    dprint_(f"\t\trival {jid} change original son {jhalo['son']} ({jhalo['son_score']:.4f}) to {-idesc} ({jdscore:.4f})", debugger)
                                                    jhalo['son'] = -jdesc
                                                    jhalo['son_score'] = jdscore
                                            else:
                                                dprint_(f"\t\trival {jid} keep original son {jhalo['son']} ({jhalo['son_score']:.4f}) rather than {-idesc} ({jdscore:.4f})", debugger)
                            # if nrival==0:
                            if ihalo['son'] == 0:
                                dprint_(f"\t{iid} change original son {ihalo['son']} ({ihalo['son_score']:.4f}) to {idesc} ({idscore:.4f})", debugger)
                                ihalo['son'] = idesc
                                ihalo['son_score'] = idscore
                            else:
                                if ihalo['son_score'] > idscore:
                                    dprint_(f"\t {iid} keep original son {ihalo['son']} ({ihalo['son_score']:.4f}) rather than {idesc} ({idscore:.4f})", debugger)
                                else:
                                    dprint_(f"\t{iid} change original son {ihalo['son']} ({ihalo['son_score']:.4f}) to {idesc} ({idscore:.4f})", debugger)
                                    ihalo['son'] = idesc
                                    ihalo['son_score'] = idscore
                            if dhalo['fat'] == 0:
                                dprint_(f"\tAlso, son {idesc} have father {iid} with {pscore:.4f}", debugger)
                                dhalo['fat'] = iid
                                dhalo['fat_score'] = pscore
                            else:
                                pass
                        else:
                            dprint_(f"\thave desc {idesc}, but his prog is {prog}", debugger)
                else:
                    do[ihalo['id']-1] = False

    dprint_("Connect same Last...", debugger, level='info')
    for iout in tqdm(nout):
        gals = inst[iout]
        for gal in gals:
            if gal['last'] == 0:
                last = gal2id(gal)
            else:
                last = gal['last']

            if (gal['son'] != 0):
                desc = inst[np.abs(gal['son'])//100000][np.abs(gal['son'])%100000 - 1]
                last = desc['last']
                # gal['last'] = last

            if (gal['fat']>0):
                prog = inst[gal['fat']//100000][gal['fat']%100000 - 1]
                if np.abs(prog['son']) == gal2id(gal):
                    prog['last'] = last
            gal['last'] = last
    dprint_("Connect same From...", debugger, level='info')
    for iout in tqdm(nout[::-1]):
        gals = inst[iout]
        for gal in gals:
            if gal['from'] == 0:
                From = gal2id(gal)
            else:
                From = gal['from']

            if (gal['fat'] != 0):
                prog = inst[np.abs(gal['fat'])//100000][np.abs(gal['fat'])%100000 - 1]
                From = prog['from']
                # gal['from'] = From

            if (gal['son']>0):
                desc = inst[gal['son']//100000][gal['son']%100000 - 1]
                if np.abs(desc['fat']) == gal2id(gal):
                    desc['from'] = From
            gal['from'] = From
    
    dprint_("Recover catalogue...", debugger, level='info')
    gals = None
    for iout in tqdm(nout):
        iinst = inst[iout]
        gals = iinst if gals is None else np.hstack((gals, iinst))
    
    dprint_("Find fragmentation...", debugger, level='info')
    uniqs = np.unique(gals['from'])
    feedback = []
    for uniq in uniqs:
        first = gals[gals['from'] == uniq][-1]
        if len(first['prog'])>0:
            prog, score = maxprog(first)
            pgal = gethalo(prog, halos=inst)
            pfirst = gals[gals['from'] == pgal['from']][-1]
            if pgal['last'] == first['last']:
                feedback.append( (uniq, first['last'], -pfirst['from'], pfirst['last']) )
            else:
                if pfirst['last']//100000 < first['timestep']:
                    feedback.append( (uniq, first['last'], pfirst['from'], first['last']) )
                    feedback.append( (pfirst['from'], pfirst['last'], pfirst['from'], first['last']) )
                else:
                    feedback.append( (uniq, first['last'], -pfirst['from'], pfirst['last']) )
    From = np.copy(gals['from'])
    Last = np.copy(gals['last'])
    for feed in tqdm(feedback):
        From[(From==feed[0])&(Last==feed[1])] = feed[2]
        Last[(From==feed[0])&(Last==feed[1])] = feed[3]
    gals['from'] = From
    gals['last'] = Last
    pklsave(gals, f"{yrepo}/ytree_stable.pickle", overwrite=True)
    dprint_(f"`{yrepo}/ytree_stable.pickle` saved\n", debugger, level='info')
# gals = pklload(f"{yrepo}/ytree_stable.pickle")



print(f"\nDone ({time.time()-ref:.3f} sec)\n")
dprint_(f"\nDone ({time.time()-ref:.3f} sec)\n", debugger, level='info')