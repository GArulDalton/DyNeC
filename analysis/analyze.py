import os,json,numpy as np,collections
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=[json.loads(l) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results','results.jsonl'))]
METH=['ce','gce','sce','boot','apl','smallloss','dynec']
MN={'ce':'CE','gce':'GCE','sce':'SCE','boot':'Bootstrap','apl':'APL (NCE+RCE)','smallloss':'Small-loss GMM','dynec':'DyNeC (ours)'}
DS=['digits','mnist','fmnist']; DN={'digits':'UCI Digits','mnist':'MNIST','fmnist':'Fashion-MNIST'}
NZ=['sym0.2','sym0.5','asym0.4','inst0.4']; NN={'sym0.2':'Sym-20%','sym0.5':'Sym-50%','asym0.4':'Pair-40%','inst0.4':'Inst-40%'}
S={}
main=collections.defaultdict(list); aux=collections.defaultdict(list); tim=collections.defaultdict(list)
for r in R:
    if r['tag']=='main':
        main[(r['ds'],r['noise'],r['method'])].append(r['acc']*100); tim[(r['ds'],r['method'])].append(r['time'])
        if r['method'] in('dynec','smallloss'): aux[(r['ds'],r['noise'],r['method'])].append(r['aux'])
tab={}
for d in DS:
  for z in NZ:
    for m in METH:
      v=main.get((d,z,m))
      if v: tab[f'{d}|{z}|{m}']=(float(np.mean(v)),float(np.std(v)),len(v))
S['main']=tab
# detection
det={}
for d in DS:
  for z in NZ:
    a=aux.get((d,z,'dynec')); b=aux.get((d,z,'smallloss'))
    if a and b:
      det[f'{d}|{z}']={k:float(np.mean([x[k] for x in a])*100) for k in ['dynec','dyn_only','nbr_only','sel_prec','sel_rec','corr_acc_noisy']}
      det[f'{d}|{z}']['smallloss']=float(np.mean([x['smallloss'] for x in b])*100)
S['det']=det
S['time']={f'{d}|{m}':float(np.mean(v)) for (d,m),v in tim.items()}
# ablation
ab=collections.defaultdict(list)
for r in R:
    if r['tag']=='abl':
        _,ds,z,sd,n=r['key'].split('|'); ab[(ds,z,n)].append(r['acc']*100)
for (ds,z) in [('digits','sym0.5'),('digits','asym0.4'),('digits','inst0.4'),('fmnist','asym0.4')]:
    v=main.get((ds,z,'dynec'))
    if v: ab[(ds,z,'full')]=v
S['abl']={f'{a}|{z}|{n}':(float(np.mean(v)),float(np.std(v)),len(v)) for (a,z,n),v in ab.items()}
se=collections.defaultdict(list)
for r in R:
    if r['tag']=='sens':
        _,ds,z,p,sd=r['key'].split('|'); se[(z,p)].append(r['acc']*100)
for z in ['sym0.5','asym0.4']:
    v=main.get(('digits',z,'dynec'))
    if v: se[(z,'k10')]=v; se[(z,'w10')]=v
S['sens']={f'{z}|{p}':(float(np.mean(v)),float(np.std(v)),len(v)) for (z,p),v in se.items()}
cl=collections.defaultdict(list)
for r in R:
    if r['tag']=='clean': cl[r['ds']].append(r['acc']*100)
S['clean']={k:(float(np.mean(v)),float(np.std(v))) for k,v in cl.items()}
S['curve']={r['method']:r['aux']['curve'] for r in R if r['tag']=='curve'}
json.dump(S,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results','summary.json'),'w'),indent=1)
# ---- figures
if all(f'fmnist|{z}|{m}' in tab for z in NZ for m in METH):
  fig,axs=plt.subplots(1,3,figsize=(13,3.8),sharey=False)
  for ax,d in zip(axs,DS):
    x=np.arange(len(NZ)); W=0.12
    for j,m in enumerate(METH):
        mu=[tab.get(f'{d}|{z}|{m}',(0,0,0))[0] for z in NZ]; sd=[tab.get(f'{d}|{z}|{m}',(0,0,0))[1] for z in NZ]
        ax.bar(x+(j-3)*W,mu,W,yerr=sd,capsize=1.5,label=MN[m],color=plt.cm.tab10(j) if m!='dynec' else '#c0392b',edgecolor='k',lw=0.3)
    ax.set_xticks(x); ax.set_xticklabels([NN[z] for z in NZ],fontsize=8.5); ax.set_title(DN[d]); ax.set_ylim(30,100); ax.grid(axis='y',alpha=.3)
  axs[0].set_ylabel('Test accuracy (%)'); axs[-1].legend(fontsize=7.2,loc='lower left',ncol=2)
  plt.tight_layout(); plt.savefig('fig3_accuracy.png',dpi=200); plt.close()
if S['curve']:
  plt.figure(figsize=(6,3.8))
  for j,m in enumerate(['ce','gce','apl','smallloss','dynec']):
    if m in S['curve']: plt.plot(np.arange(1,len(S['curve'][m])+1),np.array(S['curve'][m])*100,label=MN[m],lw=2.2 if m=='dynec' else 1.4,color='#c0392b' if m=='dynec' else None)
  plt.axvline(5,ls='--',c='gray',lw=1); plt.text(5.3,40,'end of warm-up',fontsize=8,color='gray')
  plt.xlabel('Epoch'); plt.ylabel('Test accuracy (%)'); plt.title('Fashion-MNIST, 50% symmetric noise'); plt.legend(fontsize=8); plt.grid(alpha=.3)
  plt.tight_layout(); plt.savefig('fig5_curves.png',dpi=200); plt.close()
if det:
  keys=[k for k in det]; fig,ax=plt.subplots(figsize=(11,3.6)); x=np.arange(len(keys)); W=0.2
  for j,(f,lab,c) in enumerate([('smallloss','Small-loss GMM','#7f8c8d'),('dyn_only','Temporal view only','#e67e22'),('nbr_only','Spatial view only','#27ae60'),('dynec','DyNeC fused $w_i$','#c0392b')]):
    ax.bar(x+(j-1.5)*W,[det[k][f] for k in keys],W,label=lab,color=c,edgecolor='k',lw=.3)
  ax.set_xticks(x); ax.set_xticklabels([DN[k.split('|')[0]].replace('Fashion-MNIST','F-MNIST')+'\n'+NN[k.split('|')[1]] for k in keys],fontsize=7.5)
  ax.set_ylim(50,100); ax.set_ylabel('Noise-detection AUROC (%)'); ax.legend(fontsize=8,ncol=4,loc='lower center'); ax.grid(axis='y',alpha=.3)
  plt.tight_layout(); plt.savefig('fig4_detection.png',dpi=200); plt.close()
print(len(R),'records')
