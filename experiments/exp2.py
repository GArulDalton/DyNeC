import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','dynec'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results'))
# Additional experiments: (a) ENN data-cleaning baseline, (b) downstream transfer of cleaned/corrected labels
import core,json,time,os,numpy as np,warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
OUT='results2.jsonl'; done=set()
if os.path.exists(OUT): done={json.loads(l)['key'] for l in open(OUT)}
CFG={'digits':dict(epochs=60,warm=10,bs=64,sizes=[64,128,64,10]),'mnist':dict(epochs=25,warm=5,bs=128,sizes=[784,256,128,10]),'fmnist':dict(epochs=25,warm=5,bs=128,sizes=[784,256,128,10])}
NOISE=[('sym',0.2),('sym',0.5),('asym',0.4),('inst',0.4)]
for ds in ['digits','fmnist','mnist']:
  Xtr,ytr,Xte,yte=core.load(ds); c=CFG[ds]
  for kind,r in NOISE:
    for seed in [0,1,2]:
      key=f'{ds}|{kind}{r}|{seed}'
      if key in done: continue
      yn=core.corrupt(ytr,10,kind,r,seed,X=Xtr); D=(Xtr,yn,Xte,yte,ytr); t=time.time()
      acc_enn,a_enn=core.run(D,'enn',c['epochs'],c['warm'],c['bs'],c['sizes'],seed,10,log_auc=True,ret_labels=True)
      acc_dy,a_dy=core.run(D,'dynec',c['epochs'],c['warm'],c['bs'],c['sizes'],seed,10,ret_labels=True)
      ycor=np.array(a_dy['labels']); sel=np.array(a_enn['sel'])
      def lr(X,y): return float((LogisticRegression(max_iter=150,C=0.1).fit(X,y).predict(Xte)==yte).mean())
      rec=dict(key=key,ds=ds,noise=f'{kind}{r}',seed=seed,acc_enn=acc_enn,acc_dynec_rerun=acc_dy,enn=dict((k,v) for k,v in a_enn.items() if k not in('labels','sel')),
               corr_label_acc=float((ycor==ytr).mean()),noisy_label_acc=float((yn==ytr).mean()),enn_kept_label_acc=float((yn[sel]==ytr[sel]).mean()),
               lr_noisy=lr(Xtr,yn),lr_enn=lr(Xtr[sel],yn[sel]),lr_dynec=lr(Xtr,ycor),lr_clean=lr(Xtr,ytr),time=time.time()-t)
      open(OUT,'a').write(json.dumps(rec)+'\n'); print(key,rec['acc_enn'],rec['acc_dynec_rerun'],rec['lr_noisy'],rec['lr_enn'],rec['lr_dynec'],round(rec['time']),flush=True)
print('ALLDONE',flush=True)
