from exps import *
out={}
for T,n,lab in [(1,40,'fixres'),(2,80,'fixres'),(4,160,'fixres'),(8,320,'fixres'),(2,40,'fix40'),(4,40,'fix40'),(8,40,'fix40')]:
    t=T*(np.arange(1,n+1)/n)**2; u=U[0]
    k,c=se_kappa(u,t); ek,ep=se_run(u,t,T,EPS)
    out[f'{lab}_{T}']=dict(kappa=k,cos=c,EK=ms(ek),Ep=ms(ep)); print(lab,T,out[f'{lab}_{T}'],flush=True)
json.dump(out,open('resD.json','w'),indent=1)
