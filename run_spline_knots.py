from exps import *
out={}; t=tgrid(40)
for n in [8,15,25]:
    for sp in ['uniform','quadratic']:
        e0=spline_run(n,sp,U[0],t,0); e1=spline_run(n,sp,U[0],t,EPS)
        out[f'{n}{sp}']=[float(e0[0]),ms(e1)]; print(n,sp,out[f'{n}{sp}'],flush=True)
json.dump(out,open('resE.json','w'),indent=1)
