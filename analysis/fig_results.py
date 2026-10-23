import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','figures'))
import sys; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__))))
from figstyle import *
import json,numpy as np
S=json.load(open('../results/summary.json')); tab=S['main']
METH=['ce','gce','sce','boot','apl','smallloss','dynec']
MN={'ce':'CE','gce':'GCE','sce':'SCE','boot':'Bootstrap','apl':'APL','smallloss':'Small loss','dynec':'DyNeC'}
DS=['digits','mnist','fmnist']; DN={'digits':'UCI Digits','mnist':'MNIST','fmnist':'Fashion MNIST'}
NZ=['sym0.2','sym0.5','asym0.4','inst0.4']; NN={'sym0.2':'Sym 20%','sym0.5':'Sym 50%','asym0.4':'Pair 40%','inst0.4':'Inst 40%'}
H=['','//','\\\\','xx','..','--','']; G=['#ffffff','#d9d9d9','#bdbdbd','#f0f0f0','#a6a6a6','#7f7f7f','#252525']
fig,axs=plt.subplots(3,1,figsize=(6.85,6.6))
for ax,d in zip(axs,DS):
    x=np.arange(4); W=0.115
    for j,m in enumerate(METH):
        mu=[tab[f'{d}|{z}|{m}'][0] for z in NZ]; sd=[tab[f'{d}|{z}|{m}'][1] for z in NZ]
        ax.bar(x+(j-3)*W,mu,W,yerr=sd,capsize=1.2,error_kw=dict(lw=0.6),label=MN[m],color=G[j],hatch=H[j],edgecolor='k',lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels([NN[z] for z in NZ]); ax.set_ylim(10,100); ax.set_ylabel('Test accuracy (%)')
    ax.text(0.005,0.97,f'({"abc"[DS.index(d)]}) {DN[d]}',transform=ax.transAxes,va='top',fontsize=8); ax.grid(axis='y',alpha=.3,lw=.5)
axs[0].legend(ncol=7,loc='lower center',bbox_to_anchor=(0.5,1.02),frameon=False,fontsize=7)
fig.tight_layout(); save(fig,'Fig4')
det=S['det']; keys=list(det)
fig,ax=plt.subplots(figsize=(6.85,2.6)); x=np.arange(len(keys)); W=0.2
for j,(f,lab,c,h) in enumerate([('smallloss','Small loss','#ffffff','//'),('dyn_only','Temporal view','#bdbdbd',''),('nbr_only','Spatial view','#ffffff','..'),('dynec','Fused posterior','#404040','')]):
    ax.bar(x+(j-1.5)*W,[det[k][f] for k in keys],W,label=lab,color=c,hatch=h,edgecolor='k',lw=.4)
ax.set_xticks(x); ax.set_xticklabels([{'digits':'Digits','mnist':'MNIST','fmnist':'F MNIST'}[k.split('|')[0]]+'\n'+NN[k.split('|')[1]] for k in keys],fontsize=6.5)
ax.set_ylim(50,100); ax.set_ylabel('Detection AUROC (%)'); ax.legend(ncol=4,fontsize=7,loc='lower center',bbox_to_anchor=(0.5,1.0),frameon=False); ax.grid(axis='y',alpha=.3,lw=.5)
fig.tight_layout(); save(fig,'Fig5')
cv=S['curve']; fig,ax=plt.subplots(figsize=(4.6,2.9))
for m,(ls,mk) in zip(['ce','gce','apl','smallloss','dynec'],[('-','o'),('--','s'),(':','^'),('-.','v'),('-','D')]):
    v=np.array(cv[m])*100; ax.plot(np.arange(1,len(v)+1),v,ls=ls,marker=mk,markevery=4,ms=3.5,lw=2.0 if m=='dynec' else 1.1,color='k' if m=='dynec' else '#555',label=MN[m])
ax.axvline(5,ls=':',c='gray',lw=.8); ax.text(5.4,45,'end of warm up',fontsize=7,color='#444')
ax.set_xlabel('Epoch'); ax.set_ylabel('Test accuracy (%)'); ax.legend(fontsize=7,frameon=False); ax.grid(alpha=.3,lw=.5)
fig.tight_layout(); save(fig,'Fig6')
