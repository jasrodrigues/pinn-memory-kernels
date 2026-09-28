from exps import *
out={}; t=tgrid(40)
b=exact_b(WT,LT,[U[0]],[t])
for eps in [0,1e-4,1e-3,1e-2]:
    rng=np.random.default_rng(11); rows=[]
    for d in ([0] if eps==0 else range(ND)):
        c=rng.standard_normal(4); 
        zeta=lambda s,c=c: (c[0]+c[1]*np.sin(np.pi*s)+c[2]*np.cos(np.pi*s)+c[3]*np.sin(2*np.pi*s))/np.sqrt(np.sum(c**2))
        up=lambda s,z=zeta: (1+s)*(1+eps*z(s))
        smin,k=kappa_prony([up],[t],WT,LT)
        w,l=varpro(3,[up],[t],b,nrest=10,seed=d)
        rows.append([smin,k,EKp(w,l,WT,LT)*100,Epar(w,l)*100])
    rows=np.array(rows); out[str(eps)]=[ms(rows[:,i]) for i in range(4)]; print(eps,out[str(eps)],flush=True)
json.dump(out,open('resC.json','w'),indent=1)
