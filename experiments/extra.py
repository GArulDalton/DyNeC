import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','dynec'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results'))
import core,json
CFG={'digits':dict(epochs=60,bs=64,sizes=[64,128,64,10]),'mnist':dict(epochs=25,bs=128,sizes=[784,256,128,10]),'fmnist':dict(epochs=25,bs=128,sizes=[784,256,128,10])}
for ds,c in CFG.items():
    Xtr,ytr,Xte,yte=core.load(ds)
    for s in [0,1,2]:
        acc,_=core.run((Xtr,ytr,Xte,yte,ytr),'ce',c['epochs'],5,c['bs'],c['sizes'],s,10)
        open('results.jsonl','a').write(json.dumps(dict(key=f'clean|{ds}|{s}',tag='clean',ds=ds,noise='none',seed=s,method='ce',acc=acc,aux={},time=0))+'\n')
print('EXTRADONE')
