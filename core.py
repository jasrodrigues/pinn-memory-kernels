import numpy as np
from numpy.polynomial.legendre import leggauss
xs,ws=leggauss(50); S=(xs+1)/2; WS=ws/2
def tgrid(n=40,T=1.0,zero=False):
    i=np.arange(0 if zero else 1,n+1) if not zero else np.arange(n)
    return T*(np.linspace(0,1,n+1)[1:]**2) if not zero else T*np.linspace(0,1,n)**2
def conv(Kfun,ut,t):
    # b(t)=t*int_0^1 K(t(1-s)) ut(st) ds
    tt=t[:,None]; return (tt[:,0]*(Kfun(tt*(1-S))*ut(tt*S)*WS).sum(1))
def prony(p,M):
    w=p[:M]; lam=p[M:]
    return lambda s: (w*np.exp(-np.multiply.outer(s,lam))).sum(-1)
def jac(fun,p,h=1e-7):
    f0=fun(p); J=np.zeros((f0.size,p.size))
    for k in range(p.size):
        dp=np.zeros_like(p); d=h*max(1,abs(p[k])); dp[k]=d
        J[:,k]=(fun(p+dp)-fun(p-dp))/(2*d)
    return J
def cn(J):
    Jn=J/np.linalg.norm(J,axis=0); s=np.linalg.svd(Jn,compute_uv=False); return s[0]/s[-1],s,Jn
def cosang(J,i,j): return J[:,i]@J[:,j]/np.linalg.norm(J[:,i])/np.linalg.norm(J[:,j])
def jac_prony(w,lam,ut,t):
    cols=[]
    for j in range(len(w)): cols.append(conv(lambda s,l=lam[j]: np.exp(-l*s),ut,t))
    for j in range(len(w)): cols.append(conv(lambda s,l=lam[j],ww=w[j]: -ww*s*np.exp(-l*s),ut,t))
    return np.array(cols).T
