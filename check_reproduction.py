from core import *
ut=lambda s:1+s
t=tgrid(40)
for r in [1.2,1.5,2,3,4,5,7,10]:
    k,s,Jn=cn(jac_prony([1,1,1],[1,r,r*r],ut,t))
    print(r,'%.2g'%k,round(cosang(Jn,3,4),4),round(cosang(Jn,3,5),4))
for zero in [True,False]:
    k,s,Jn=cn(jac_prony([1,.6,.3],[.5,2,8],ut,tgrid(40,zero=zero))); print('zero',zero,'%.4g'%k,'%.3g'%s[-1])
se=lambda p: (lambda s: p[0]*np.exp(-(np.maximum(s,0)/p[1])**p[2]))
for T in [1,2,3,5,10]:
    tt=tgrid(40,T=T); p=np.array([1.9,.55,.7])
    k,s,Jn=cn(jac(lambda q: conv(se(q),ut,tt),p,h=1e-6))
    print('SE',T,round(k,2),round(cosang(Jn,0,1),3),round(cosang(Jn,0,2),3),round(cosang(Jn,1,2),3))
