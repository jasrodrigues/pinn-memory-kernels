from exps import *
out={}
for nt,n in [(1,240),(2,120),(4,60)]:
    t=(np.arange(1,n+1)/n)**2; uts=U[:nt]; ts=[t]*nt
    smin,k=kappa_prony(uts,ts,WT,LT); d=dict(smin=smin,k=k)
    for M in [3,12]:
        ek,ep=prony_run(M,uts,ts,WT,LT,EPS); d[M]=[ms(ek),ms(ep)]
    out[f'{nt}x{n}']=d; print(nt,n,d,flush=True)
json.dump(out,open('resG.json','w'),indent=1)
