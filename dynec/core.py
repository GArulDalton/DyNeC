import numpy as np, gzip, warnings
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
import os
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data')+os.sep
def _idx(f,img):
    with gzip.open(f) as g: b=g.read()
    return np.frombuffer(b,np.uint8,offset=16).reshape(-1,784).astype(np.float32)/255. if img else np.frombuffer(b,np.uint8,offset=8).astype(np.int64)
def load(name,ntr=20000,seed=0):
    if name=='digits':
        d=load_digits(); X=d.data.astype(np.float32)/16.; y=d.target
        r=np.random.RandomState(123).permutation(len(y)); X,y=X[r],y[r]; n=1257
        Xtr,ytr,Xte,yte=X[:n],y[:n],X[n:],y[n:]
    else:
        p={'mnist':'mn_','fmnist':'fm_'}[name]
        Xtr=_idx(D+p+'train-images-idx3-ubyte.gz',1); ytr=_idx(D+p+'train-labels-idx1-ubyte.gz',0)
        Xte=_idx(D+p+'t10k-images-idx3-ubyte.gz',1); yte=_idx(D+p+'t10k-labels-idx1-ubyte.gz',0)
        r=np.random.RandomState(123).permutation(len(ytr))[:ntr]; Xtr,ytr=Xtr[r],ytr[r]
    m,s=Xtr.mean(0),Xtr.std(0)+1e-3
    return (Xtr-m)/s,ytr,(Xte-m)/s,yte
def corrupt(y,C,kind,rate,seed,X=None):
    rng=np.random.RandomState(seed); yn=y.copy(); flip=rng.rand(len(y))<rate
    if kind=='sym':
        for i in np.where(flip)[0]:
            yn[i]=rng.choice([c for c in range(C) if c!=y[i]])
    elif kind=='asym':   # pair flip c -> c+1
        yn[flip]=(y[flip]+1)%C
    elif kind=='inst':   # instance-dependent: flip to the most confusable class by a linear proxy
        from sklearn.linear_model import LogisticRegression
        lr=LogisticRegression(max_iter=200,C=0.05).fit(X,y); P=lr.predict_proba(X); P[np.arange(len(y)),y]=-1
        hard=P.max(1); tgt=P.argmax(1); order=np.argsort(-hard); k=int(rate*len(y)); sel=order[:k]
        yn[sel]=tgt[sel]
    return yn
Xglob=None
def softmax(z):
    z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
class MLP:
    def __init__(s,sizes,rng,lr=1e-3,wd=5e-4):
        s.W=[rng.randn(a,b).astype(np.float32)*np.sqrt(2/a) for a,b in zip(sizes[:-1],sizes[1:])]
        s.b=[np.zeros(b,np.float32) for b in sizes[1:]]
        s.P=s.W+s.b; s.m=[np.zeros_like(p) for p in s.P]; s.v=[np.zeros_like(p) for p in s.P]; s.t=0; s.lr=lr; s.wd=wd
    def fwd(s,X):
        A=[X]; h=X
        for i,(W,b) in enumerate(zip(s.W,s.b)):
            h=h@W+b
            if i<len(s.W)-1: h=np.maximum(h,0)
            A.append(h)
        return h,A
    def bwd(s,A,dz):
        gW=[None]*len(s.W); gb=[None]*len(s.W)
        for i in range(len(s.W)-1,-1,-1):
            gW[i]=A[i].T@dz+s.wd*s.W[i]; gb[i]=dz.sum(0)
            if i>0: dz=(dz@s.W[i].T)*(A[i]>0)
        s.t+=1; b1,b2=0.9,0.999
        for j,(p,g) in enumerate(zip(s.P,gW+gb)):
            s.m[j]=b1*s.m[j]+(1-b1)*g; s.v[j]=b2*s.v[j]+(1-b2)*g*g
            p-=s.lr*(s.m[j]/(1-b1**s.t))/(np.sqrt(s.v[j]/(1-b2**s.t))+1e-8)
    def predict(s,X,bs=4096):
        out=[];emb=[]
        for i in range(0,len(X),bs):
            z,A=s.fwd(X[i:i+bs]); out.append(softmax(z)); emb.append(A[-2])
        return np.vstack(out),np.vstack(emb)
def gmm_clean(v,seed=0):
    v=np.asarray(v,np.float64).reshape(-1,1)
    if len(v)<10 or v.std()<1e-6: return np.ones(len(v))
    g=GaussianMixture(2,reg_covar=1e-3,max_iter=50,random_state=seed).fit(v)
    return g.predict_proba(v)[:,np.argmax(g.means_.ravel())]
def knn_graph(E,k,bs=2048):
    E=E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-8); idx=np.zeros((len(E),k),np.int64)
    for i in range(0,len(E),bs):
        S=E[i:i+bs]@E.T; S[np.arange(S.shape[0]),np.arange(i,i+S.shape[0])]=-np.inf
        idx[i:i+bs]=np.argpartition(-S,k,axis=1)[:,:k]
    return idx
def logit(p): p=np.clip(p,1e-4,1-1e-4); return np.log(p/(1-p))
def run(data,method,epochs,warm,bs,sizes,seed,C,lr=1e-3,k=10,flags=None,log_auc=False,curve=False,ret_labels=False):
    run.nb=None; cur=[]
    """method in ce,gce,sce,boot,elr,smallloss,dynec. flags: ablations for dynec"""
    fl=dict(dyn=True,nbr=True,classcond=True,correct=True,relw=True); fl.update(flags or {})
    Xtr,yn,Xte,yte,ytrue=data; n=len(yn); rng=np.random.RandomState(seed)
    net=MLP(sizes,rng,lr=lr); Y=np.eye(C,dtype=np.float32)[yn]
    pbar=np.full((n,C),1./C,np.float32); msum=np.zeros(n); mcnt=0
    w=np.ones(n); T=Y.copy(); sel=np.ones(n,bool); aucs={}
    if method=='enn':   # edited nearest neighbour filter (Wilson 1972) on standardised inputs
        nbr=knn_graph(Xtr,k); votes=np.eye(C)[yn[nbr]].sum(1); agree=votes[np.arange(n),yn]/k
        sel=votes.argmax(1)==yn
        if log_auc: aucs['enn_auc']=roc_auc_score(yn==ytrue,agree); aucs['sel_prec']=float((yn==ytrue)[sel].mean()); aucs['sel_rec']=float(sel[yn==ytrue].mean())
        aucs['kept']=float(sel.mean())
    for ep in range(epochs):
        # --- per-epoch refinement ---
        if method=='smallloss' and ep>=warm:
            P,_=net.predict(Xtr); loss=-np.log(P[np.arange(n),yn]+1e-8)
            w=gmm_clean(-loss,seed); sel=w>0.5
            if log_auc and ep==epochs-1: aucs['smallloss']=roc_auc_score(yn==ytrue,w)
        if method=='dynec' and ep>=warm:
            P,E=net.predict(Xtr)
            pd=np.ones(n); pn=np.ones(n); prior=np.full(n,0.5)
            groups=[np.where(yn==c)[0] for c in range(C)] if fl['classcond'] else [np.arange(n)]
            if fl['nbr'] and (ep-warm)%3==0: 
                nb=knn_graph(E,k); run.nb=nb
            nb=run.nb if fl['nbr'] else None
            if fl['nbr']:
                wn=w[nb] if fl['relw'] else np.ones_like(nb,dtype=float)
                agree=(wn*(yn[nb]==yn[:,None])).sum(1)/(wn.sum(1)+1e-8)
                Q=np.einsum('ik,ikc->ic',wn,np.eye(C)[yn[nb]])/(wn.sum(1,keepdims=True)+1e-8)
            mu=msum/max(mcnt,1)
            for g in groups:
                if fl['dyn']: pd[g]=gmm_clean(mu[g],seed)
                if fl['nbr']: pn[g]=gmm_clean(agree[g],seed)
                prior[g]=np.clip(pd[g].mean() if fl['dyn'] else pn[g].mean(),0.05,0.95)
            if fl.get('fusion','poe')=='joint' and fl['dyn'] and fl['nbr']:
                for g in groups:
                    Z=np.c_[mu[g],agree[g]]; Z=(Z-Z.mean(0))/(Z.std(0)+1e-6)
                    gm=GaussianMixture(2,covariance_type='full',reg_covar=1e-3,max_iter=50,random_state=seed).fit(Z)
                    pd[g]=gm.predict_proba(Z)[:,np.argmax(gm.means_.sum(1))]; pn[g]=pd[g]; prior[g]=0.5
                L=logit(pd)
            else:
              L=np.zeros(n)
              if fl['dyn']: L+=logit(pd)
              if fl['nbr']: L+=logit(pn)
              if fl['dyn'] and fl['nbr']: L-=logit(prior)
            w=1/(1+np.exp(-L))
            if fl['correct']:
                base=0.5*pbar+0.5*Q if fl['nbr'] else pbar
                sh=base**2; sh/=sh.sum(1,keepdims=True)
                T=(w[:,None]*Y+(1-w[:,None])*sh).astype(np.float32); sel=np.ones(n,bool)
            else:
                T=Y; sel=w>0.5
            if log_auc and ep==epochs-1:
                clean=(yn==ytrue); aucs['dynec']=roc_auc_score(clean,w)
                if fl['dyn']: aucs['dyn_only']=roc_auc_score(clean,mu)
                if fl['nbr']: aucs['nbr_only']=roc_auc_score(clean,agree)
                aucs['sel_prec']=float(clean[w>0.5].mean()); aucs['sel_rec']=float((w>0.5)[clean].mean())
                aucs['corr_acc_noisy']=float((T.argmax(1)[~clean]==ytrue[~clean]).mean())
        # --- minibatch training ---
        perm=rng.permutation(np.where(sel)[0])
        for i in range(0,len(perm),bs):
            b=perm[i:i+bs]; z,A=net.fwd(Xtr[b]); p=softmax(z); yb=yn[b]; B=len(b)
            if method in('dynec',) and ep>=warm:
                g=(p-T[b])
            elif method=='gce':
                py=p[np.arange(B),yb][:,None]; oh=Y[b]; g=-(py**0.7)*(oh-p)
            elif method=='sce':
                py=p[np.arange(B),yb][:,None]; oh=Y[b]; g=0.1*(p-oh)-4.0*py*(oh-p)
            elif method=='boot' and ep>=warm:
                t=0.8*Y[b]+0.2*np.eye(C,dtype=np.float32)[p.argmax(1)]; g=p-t
            elif method=='apl':
                oh=Y[b]; lp=np.log(np.clip(p,1e-7,1)); A_=-lp[np.arange(B),yb][:,None]; S_=-lp.sum(1,keepdims=True)
                gN=((p-oh)*S_-A_*(C*p-1))/S_**2; py=p[np.arange(B),yb][:,None]
                g=1.0*gN-1.0*4.0*py*(oh-p)
            elif method=='elr':
                pc=np.clip(p,1e-4,1-1e-4); pc=pc/pc.sum(1,keepdims=True)
                pbar[b]=0.7*pbar[b]+0.3*pc; tp=(pbar[b]*pc).sum(1,keepdims=True)
                gp=-3.0*pbar[b]/(1-tp); g=(p-Y[b])+pc*(gp-(gp*pc).sum(1,keepdims=True))
            else:
                g=p-Y[b]
            if method=='dynec':
                pbar[b]=0.7*pbar[b]+0.3*p
                srt=np.sort(p,1); py=p[np.arange(B),yb]; other=np.where(np.isclose(srt[:,-1],py),srt[:,-2],srt[:,-1])
                msum[b]+=py-other
            net.bwd(A,(g/B).astype(np.float32))
        if method=='dynec': mcnt+=1
        if curve: cur.append(float((net.predict(Xte)[0].argmax(1)==yte).mean()))
    P,_=net.predict(Xte); acc=float((P.argmax(1)==yte).mean())
    if curve: aucs['curve']=cur
    if ret_labels: aucs['labels']=(T.argmax(1) if method=='dynec' else yn).tolist(); aucs['sel']=sel.tolist()
    return acc,aucs
run.nb=None
