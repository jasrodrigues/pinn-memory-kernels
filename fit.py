from core import *
from scipy.optimize import nnls, least_squares, linear_sum_assignment
WT,LT=np.array([1,.6,.3]),np.array([.5,2,8])
def design(lam,uts,ts):
    return np.vstack([np.array([conv(lambda s,l=l: np.exp(-l*s),ut,t) for l in lam]).T for ut,t in zip(uts,ts)])
def exact_b(w,lam,uts,ts):
    # closed-form-free: high-order GL (200 nodes) is exact to machine precision for these integrands
    from numpy.polynomial.legendre import leggauss
    x,wq=leggauss(200); s=(x+1)/2; wq=wq/2
    out=[]
    for ut,t in zip(uts,ts):
        tt=t[:,None]; K=(w*np.exp(-np.multiply.outer(tt*(1-s),lam))).sum(-1)
        out.append(tt[:,0]*(K*ut(tt*s)*wq).sum(1))
    return np.concatenate(out)
def solve_w(Phi,b,alpha):
    if alpha>0:
        A=np.vstack([Phi,np.sqrt(alpha)*np.diag(np.linalg.norm(Phi,axis=0))]); bb=np.r_[b,np.zeros(Phi.shape[1])]
    else: A,bb=Phi,b
    w,_=nnls(A,bb,maxiter=5000); return w
def varpro(M,uts,ts,b,alpha=0.,nrest=20,seed=0,lmax=20.):
    rng=np.random.default_rng(seed); best=None
    for r in range(nrest):
        th0=np.sort(rng.uniform(np.log(1e-3),np.log(lmax),M))
        def res(th):
            Phi=design(np.exp(th),uts,ts); w=solve_w(Phi,b,alpha)
            rr=Phi@w-b
            if alpha>0: rr=np.r_[rr,np.sqrt(alpha)*np.linalg.norm(Phi,axis=0)*w]
            return rr
        sol=least_squares(res,th0,bounds=(np.log(1e-3),np.log(lmax)),x_scale=1.0,max_nfev=400,xtol=1e-14,ftol=1e-14,gtol=1e-14)
        lam=np.exp(sol.x); w=solve_w(design(lam,uts,ts),b,alpha)
        c=np.sum(sol.fun**2)
        if best is None or c<best[0]: best=(c,w,lam)
    return best[1],best[2]
TE=np.linspace(0,1,1001)
def EK(w,lam,wt=WT,lt=LT,te=TE):
    K=(w*np.exp(-np.multiply.outer(te,lam))).sum(-1); Kt=(wt*np.exp(-np.multiply.outer(te,lt))).sum(-1)
    return np.sqrt(np.trapezoid((K-Kt)**2,te)/np.trapezoid(Kt**2,te))
def Epar(w,lam,wt=WT,lt=LT):
    k=len(wt); idx=np.argsort(w)[::-1][:k]; w2,l2=w[idx],lam[idx]
    C=((l2[:,None]-lt[None])/lt[None])**2+((w2[:,None]-wt[None])/wt[None])**2
    i,j=linear_sum_assignment(C)
    p=np.r_[l2[i],w2[i]]; q=np.r_[lt[j],wt[j]]
    return np.linalg.norm(p-q)/np.linalg.norm(q)
