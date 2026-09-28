import numpy as np, json
from numpy.polynomial.legendre import leggauss
from scipy.optimize import nnls, least_squares, linear_sum_assignment
from exps import EKp, noisy
from fit import WT,LT,Epar
from core import cn
u=lambda s:1+s
def rule(n):
    x,w=leggauss(n); return (x+1)/2,w/2
def design(lam,t,n):
    S,W=rule(n); tt=t[:,None]
    return np.array([tt[:,0]*(np.exp(-l*tt*(1-S))*u(tt*S)*W).sum(1) for l in lam]).T
def jacobian(t,n):
    S,W=rule(n); tt=t[:,None]; cols=[]
    for l in LT: cols.append(tt[:,0]*(np.exp(-l*tt*(1-S))*u(tt*S)*W).sum(1))
    for l,w in zip(LT,WT): cols.append(tt[:,0]*(-w*tt*(1-S)*np.exp(-l*tt*(1-S))*u(tt*S)*W).sum(1))
    return np.array(cols).T
def target(t):
    return design(LT,t,200)@WT
def fitM3(t,n,b,seed,nrest=6):
    rng=np.random.default_rng(seed); best=None
    for r in range(nrest):
        th0=np.sort(rng.uniform(np.log(1e-3),np.log(20),3))
        def res(th):
            P=design(np.exp(th),t,n); w,_=nnls(P,b); return P@w-b
        sol=least_squares(res,th0,bounds=(np.log(1e-3),np.log(20)),xtol=1e-14,ftol=1e-14,gtol=1e-14,max_nfev=400)
        c=np.sum(sol.fun**2)
        if best is None or c<best[0]:
            lam=np.exp(sol.x); w,_=nnls(design(lam,t,n),b); best=(c,w,lam)
    return best[1],best[2]
out={}
for Nt,Nq in [(20,50),(40,50),(80,50),(40,25),(40,100)]:
    t=(np.arange(1,Nt+1)/Nt)**2; b=target(t)
    k=cn(jacobian(t,Nq))[0]
    w0,l0=fitM3(t,Nq,b,0,nrest=20); ex=[EKp(w0,l0,WT,LT)*100,Epar(w0,l0)*100]
    rng=np.random.default_rng(21); ek=[];ep=[]
    for d in range(10):
        w,l=fitM3(t,Nq,noisy(b,rng),100+d); ek.append(EKp(w,l,WT,LT)*100); ep.append(Epar(w,l)*100)
    out[f'{Nt},{Nq}']=dict(kappa=k,exact=ex,EK=[np.mean(ek),np.std(ek)],Ep=[np.mean(ep),np.std(ep)])
    print(Nt,Nq,out[f'{Nt},{Nq}'],flush=True)
json.dump(out,open('resDisc.json','w'),indent=1,default=float)
