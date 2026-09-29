import numpy as np, json
from exps import frac_conv, EKd, te, noisy, ND
from fit import varpro, tgrid
from scipy.special import gamma
from scipy.optimize import minimize_scalar
u=lambda s:1+s; t=tgrid(40); a=0.6
b=frac_conv(a,u,t,200)
tt=te(); Kt=tt**(-a)/gamma(1-a)
out={}
def prony_eval(w,l): return (w*np.exp(-np.multiply.outer(tt,l))).sum(-1)
for lmax in [20.,1e3]:
    for eps in [0,1e-3]:
        rng=np.random.default_rng(5); r=[]
        for d in ([0] if eps==0 else range(ND)):
            bb=b if eps==0 else noisy(b,rng,eps)
            w,l=varpro(12,[u],[t],bb,nrest=(20 if eps==0 else 6),seed=d,lmax=lmax)
            from fit import design
            res=np.linalg.norm(design(l,[u],[t])@w-bb)/np.linalg.norm(bb)
            K=prony_eval(w,l)
            r.append([EKd(K,Kt,tt)*100,EKd(K,Kt,tt,True)*100,res*100,int((w>1e-8*w.max()).sum())])
        r=np.array(r); out[f'prony_lmax{lmax:g}_eps{eps}']=[r.mean(0).tolist(),r.std(0).tolist()]
        print('prony',lmax,eps,np.round(r.mean(0),4),np.round(r.std(0),4),flush=True)
# direct alpha
for eps in [0,1e-3]:
    rng=np.random.default_rng(5); r=[]
    for d in ([0] if eps==0 else range(ND)):
        bb=b if eps==0 else noisy(b,rng,eps)
        sol=minimize_scalar(lambda al: np.sum((frac_conv(al,u,t,50)-bb)**2),bounds=(0.01,0.99),method='bounded',options={'xatol':1e-12})
        ah=sol.x; K=tt**(-ah)/gamma(1-ah)
        res=np.linalg.norm(frac_conv(ah,u,t,50)-bb)/np.linalg.norm(bb)
        r.append([EKd(K,Kt,tt)*100,EKd(K,Kt,tt,True)*100,res*100,abs(ah-a)/a*100])
    r=np.array(r); out[f'alpha_eps{eps}']=[r.mean(0).tolist(),r.std(0).tolist()]
    print('alpha',eps,np.round(r.mean(0),5),np.round(r.std(0),5),flush=True)
json.dump(out,open('resFracProny.json','w'),indent=1)
