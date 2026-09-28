from exps import *
out={}
t=tgrid(40); u=[U[0]]
# E1: M study
for M in [2,3,6,12]:
    e0k,e0p=prony_run(M,u,[t],WT,LT,0)
    ek,ep=prony_run(M,u,[t],WT,LT,EPS)
    out[f'M{M}']=dict(exact=[float(e0k[0]),float(e0p[0])],noisy=[ms(ek),ms(ep)])
    print(M,out[f'M{M}'],flush=True)
json.dump(out,open('resA.json','w'),indent=1)
