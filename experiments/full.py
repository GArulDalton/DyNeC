import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','dynec'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results'))
import core,json,time,os
OUT='results.jsonl'
done=set()
if os.path.exists(OUT):
    for l in open(OUT): d=json.loads(l); done.add(d['key'])
CFG={'digits':dict(epochs=60,warm=10,bs=64,sizes=[64,128,64,10]),
     'mnist':dict(epochs=25,warm=5,bs=128,sizes=[784,256,128,10]),
     'fmnist':dict(epochs=25,warm=5,bs=128,sizes=[784,256,128,10])}
NOISE=[('sym',0.2),('sym',0.5),('asym',0.4),('inst',0.4)]
METH=['ce','gce','sce','boot','apl','smallloss','dynec']
cache={}
def data(ds,kind,r,seed):
    if ds not in cache: cache[ds]=core.load(ds)
    Xtr,ytr,Xte,yte=cache[ds]; yn=core.corrupt(ytr,10,kind,r,seed,X=Xtr)
    return (Xtr,yn,Xte,yte,ytr)
def do(key,ds,kind,r,seed,m,flags=None,k=10,warm=None,curve=False,tag='main'):
    if key in done: return
    c=dict(CFG[ds]); 
    if warm is not None: c['warm']=warm
    t=time.time(); acc,a=core.run(data(ds,kind,r,seed),m,c['epochs'],c['warm'],c['bs'],c['sizes'],seed,10,k=k,flags=flags,log_auc=True,curve=curve)
    rec=dict(key=key,tag=tag,ds=ds,noise=f'{kind}{r}',seed=seed,method=m,flags=flags,k=k,warm=c['warm'],acc=acc,aux=a,time=time.time()-t)
    open(OUT,'a').write(json.dumps(rec)+'\n'); print(key,acc,round(rec['time']),flush=True)
# 1 main
for ds in ['digits','mnist','fmnist']:
  for kind,r in NOISE:
    for seed in [0,1,2]:
      for m in METH: do(f'main|{ds}|{kind}{r}|{seed}|{m}',ds,kind,r,seed,m)
# 2 ablation
AB={'full':{},'no_dyn':{'dyn':False},'no_nbr':{'nbr':False},'no_classcond':{'classcond':False},'no_relw':{'relw':False},'no_correct':{'correct':False},'joint_fusion':{'fusion':'joint'}}
for ds,settings in [('digits',[('sym',0.5),('asym',0.4),('inst',0.4)]),('fmnist',[('asym',0.4)])]:
 for kind,r in settings:
  for seed in [0,1,2]:
    for n,f in AB.items():
      if n=='full': continue
      do(f'abl|{ds}|{kind}{r}|{seed}|{n}',ds,kind,r,seed,'dynec',flags=f,tag='abl')
# 3 sensitivity (Digits)
for kind,r in [('sym',0.5),('asym',0.4)]:
 for seed in [0,1,2]:
  for k in [5,20]: do(f'sens|digits|{kind}{r}|k{k}|{seed}','digits',kind,r,seed,'dynec',k=k,tag='sens')
  for w in [5,20]: do(f'sens|digits|{kind}{r}|w{w}|{seed}','digits',kind,r,seed,'dynec',warm=w,tag='sens')
# 4 curves
for m in ['ce','gce','apl','smallloss','dynec']:
  do(f'curve|{m}','fmnist','sym',0.5,0,m,curve=True,tag='curve')
print('ALLDONE',flush=True)
