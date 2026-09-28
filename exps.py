import sys, json, numpy as np
from fit import *
from scipy.special import roots_jacobi, gamma
from scipy.optimize import minimize_scalar
from numpy.polynomial.legendre import leggauss
U=[lambda s:1+s, lambda s:np.ones_like(s), lambda s:np.exp(-2*s), lambda s:np.sin(2*np.pi*s), lambda s:s**2]
UG2=lambda s: np.pi*np.cos(2*np.pi*s)
EPS=1e-3; ND=10
def te(T=1.,n=200): return T*(np.arange(1,n+1)/n)**2
def EKd(K,Kt,t,weight_t=False):
    d=(K-Kt)**2; n=Kt**2
    if weight_t: d=d*t; n=n*t
    return np.sqrt(np.trapezoid(d,t)/np.trapezoid(n,t))
def pr(w,l,t): return (w*np.exp(-np.multiply.outer(t,l))).sum(-1)
def EKp(w,l,wt,lt,T=1.): t=te(T); return EKd(pr(w,l,t),pr(wt,lt,t),t)
def noisy(b,rng,eps=EPS): return b+eps*np.sqrt(np.mean(b**2))*rng.standard_normal(b.shape)
def prony_run(M,uts,ts,wt,lt,eps,nd=ND,nrest=6,seed=1,alpha=0.,lmax=20.):
    b=exact_b(wt,lt,uts,ts); rng=np.random.default_rng(seed); ek=[];ep=[]
    draws=[0] if eps==0 else range(nd)
    for d in draws:
        bb=b if eps==0 else noisy(b,rng,eps)
        w,l=varpro(M,uts,ts,bb,alpha=alpha,nrest=(20 if eps==0 else nrest),seed=seed+d,lmax=lmax)
        ek.append(EKp(w,l,wt,lt)); ep.append(Epar(w,l,wt,lt) if M>=len(wt) else np.nan)
    return np.array(ek)*100,np.array(ep)*100
def kappa_prony(uts,ts,wt,lt):
    k,s,_=cn(np.vstack([jac_prony(wt,lt,u,t) for u,t in zip(uts,ts)])); return s[-1],k
# ---------- stretched
def se_K(A,tau,beta,s): return A*np.exp(-(np.maximum(s,0)/tau)**beta)
def conv_n(Kf,ut,t,n):
    x,w=leggauss(n); S=(x+1)/2; W=w/2; tt=t[:,None]; return tt[:,0]*(Kf(tt*(1-S))*ut(tt*S)*W).sum(1)
def se_fit(ut,t,b,seed=0):
    best=None
    for tau0 in [0.2,0.6,1.5]:
        for b0 in [0.5,1.0,1.5]:
            def res(q):
                phi=conv(lambda s: se_K(1,q[0],q[1],s),ut,t); A=max(phi@b/(phi@phi),0); return A*phi-b
            sol=least_squares(res,[tau0,b0],bounds=([0.02,0.1],[10,3]),xtol=1e-14,ftol=1e-14,gtol=1e-14,max_nfev=500)
            c=np.sum(sol.fun**2)
            if best is None or c<best[0]:
                phi=conv(lambda s: se_K(1,*sol.x,s),ut,t); best=(c,max(phi@b/(phi@phi),0),*sol.x)
    return np.array(best[1:])
def se_run(ut,t,T,eps,p=np.array([1.9,.55,.7]),seed=3):
    b=conv_n(lambda s: se_K(*p,s),ut,t,400); rng=np.random.default_rng(seed); ek=[];ep=[]
    for d in ([0] if eps==0 else range(ND)):
        bb=b if eps==0 else noisy(b,rng,eps); q=se_fit(ut,t,bb)
        tt=te(T); ek.append(EKd(se_K(*q,tt),se_K(*p,tt),tt)); ep.append(np.linalg.norm(q-p)/np.linalg.norm(p))
    return np.array(ek)*100,np.array(ep)*100
def se_kappa(ut,t,p=np.array([1.9,.55,.7])):
    k,s,Jn=cn(jac(lambda q: conv(lambda s: se_K(*q,s),ut,t),p,h=1e-6)); return k,cosang(Jn,1,2)
# ---------- fractional
def frac_conv(alpha,ut,t,n):
    x,w=roots_jacobi(n,-alpha,0.)
    tt=t[:,None]; return (tt[:,0]**(1-alpha)/gamma(1-alpha))*2**(alpha-1)*(ut(tt*(x+1)/2)*w).sum(1)
def frac_run(ut,t,eps,a=0.6,seed=5):
    b=frac_conv(a,ut,t,200); rng=np.random.default_rng(seed); r=[]
    for d in ([0] if eps==0 else range(ND)):
        bb=b if eps==0 else noisy(b,rng,eps)
        sol=minimize_scalar(lambda al: np.sum((frac_conv(al,ut,t,50)-bb)**2),bounds=(0.01,0.99),method='bounded',options={'xatol':1e-12})
        ah=sol.x; tt=te(); K=lambda al: tt**(-al)/gamma(1-al)
        r.append([EKd(K(ah),K(a),tt),EKd(K(ah),K(a),tt,True),abs(ah-a)/a])
    return np.array(r)*100
# ---------- spline
def spline_design(knots,ut,t):
    n=len(knots); h=np.diff(knots)
    # v = L theta, theta=(c,a,d_1..d_{n-2}) >=0
    L=np.zeros((n,n))
    # slopes m_k for segment k=1..n-1 (index k-1): m_{n-1}=-a ; m_k=m_{k+1}-d_k
    Msl=np.zeros((n-1,n))  # slope as linear comb of theta
    Msl[n-2,1]=-1
    for k in range(n-3,-1,-1): Msl[k]=Msl[k+1]; Msl[k,2+k]-=1
    L[n-1,0]=1
    for k in range(n-2,-1,-1): L[k]=L[k+1]-Msl[k]*h[k]
    cols=[]
    for j in range(n):
        e=np.zeros(n); e[j]=1
        cols.append(conv(lambda s,e=e: np.interp(s,knots,e),ut,t))
    return np.array(cols).T@L, L
def spline_run(n,spacing,ut,t,eps,wt=WT,lt=LT,seed=7):
    k=np.linspace(0,1,n); knots=k if spacing=='uniform' else k**2
    A,L=spline_design(knots,ut,t); b=exact_b(wt,lt,[ut],[t]); rng=np.random.default_rng(seed); ek=[]
    for d in ([0] if eps==0 else range(ND)):
        bb=b if eps==0 else noisy(b,rng,eps); th,_=nnls(A,bb,maxiter=20000); v=L@th
        tt=te(); ek.append(EKd(np.interp(tt,knots,v),pr(wt,lt,tt),tt))
    return np.array(ek)*100
def ms(x): 
    x=np.asarray(x,float)
    return [float(np.nanmean(x)),float(np.nanstd(x))]
