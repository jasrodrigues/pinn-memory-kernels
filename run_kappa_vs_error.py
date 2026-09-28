from exps import *
out={}; t=tgrid(40); u=[U[0]]
for r in [1.2,1.5,2,3,4,5,7,10]:
    wt=np.array([1.,1,1]); lt=np.array([1,r,r*r]); d={}
    for M in [3,12]:
        ek,ep=prony_run(M,u,[t],wt,lt,EPS,lmax=max(20.,1.5*r*r) if r>=5 else 20.); d[M]=[ms(ek),ms(ep)]
        e0k,e0p=prony_run(M,u,[t],wt,lt,0,lmax=max(20.,1.5*r*r) if r>=5 else 20.); d[f'{M}ex']=[float(e0k[0]),float(e0p[0])]
    out[str(r)]=d; print(r,d,flush=True)
json.dump(out,open('resB.json','w'),indent=1)
