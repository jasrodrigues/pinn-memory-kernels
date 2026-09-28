from exps import *
out={}; t=tgrid(40)
for gname,u in [('g1',U[0]),('g2',UG2)]:
    d={}
    d['kP']=kappa_prony([u],[t],WT,LT)[1]; d['kS']=se_kappa(u,t)[0]
    for M in [3,12]:
        ek,ep=prony_run(M,[u],[t],WT,LT,EPS); d[f'P{M}']=[ms(ek),ms(ep)]
    fr=frac_run(u,t,EPS); d['F']=[ms(fr[:,0]),ms(fr[:,1]),ms(fr[:,2])]
    fr0=frac_run(u,t,0); d['F0']=fr0[0].tolist()
    ek,ep=se_run(u,t,1.,EPS); d['S']=[ms(ek),ms(ep)]
    e=spline_run(15,'quadratic',u,t,EPS); d['Sp']=ms(e)
    out[gname]=d; print(gname,d,flush=True)
json.dump(out,open('resF.json','w'),indent=1)
