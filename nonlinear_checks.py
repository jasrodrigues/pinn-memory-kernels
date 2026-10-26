"""
Nonlinear checks for the proceedings paper (double precision, CPU only).
  Table 3: range of validity of the first-order predictions
           (nonlinear Prony fits, M=3, eps in {1e-6,...,1e-3}, 10 draws, 10 restarts)
  Sect. 5: kappa_rep and kappa_obs evaluated at the estimates instead of the truth
  Table 4 (absolute-noise columns): excitations normalised to unit L2 norm (absolute-noise criteria)
"""
import numpy as np
from scipy.optimize import least_squares, nnls
from scipy.integrate import quad

xg, wg = np.polynomial.legendre.leggauss(50); s = (xg + 1) / 2; om = wg / 2
tg = (np.arange(1, 41) / 40)**2
kg = (np.arange(1, 201) / 200)**2
kw = np.empty(200); kw[1:-1] = (kg[2:] - kg[:-2]) / 2
kw[0] = (kg[1] - kg[0]) / 2; kw[-1] = (kg[-1] - kg[-2]) / 2

def Jmat(w, lam, gp=lambda x: 1 + x, tgrid=tg):
    T = tgrid[:, None]; A = T * (1 - s[None, :]); GP = gp(s[None, :] * T)
    B = np.stack([(T * om * np.exp(-l * A) * GP).sum(1) for l in lam], 1)
    dB = np.stack([(T * om * (-A) * np.exp(-l * A) * GP).sum(1) for l in lam], 1)
    return B, dB, np.hstack([B, dB * w])

def Gmat(w, lam):
    E = np.exp(-np.outer(kg, lam))
    return np.hstack([E, -w * kg[:, None] * E]) * np.sqrt(kw)[:, None]

def factor(J, G):
    d = np.linalg.norm(G, axis=0); U, S, Vt = np.linalg.svd(G / d, full_matrices=False)
    B = (J / d) @ Vt.T / S; sb = np.linalg.svd(B, compute_uv=False)
    return S[0] / S[-1], sb[0] / sb[-1], sb[-1], B

def fit(b, M, rng, nrest=10, lmax=20.0):
    lo = np.r_[np.zeros(M), np.full(M, np.log(1e-3))]
    hi = np.r_[np.full(M, np.inf), np.full(M, np.log(lmax))]
    def res(x): return Jmat(x[:M], np.exp(x[M:]))[0] @ x[:M] - b
    def jac(x):
        B, dB, _ = Jmat(x[:M], np.exp(x[M:])); return np.hstack([B, dB * x[:M] * np.exp(x[M:])])
    best = None
    for _ in range(nrest):
        lam0 = np.sort(np.exp(rng.uniform(np.log(1e-3), np.log(lmax), M)))
        w0, _ = nnls(Jmat(np.ones(M), lam0)[0], b)
        x0 = np.clip(np.r_[w0 + 1e-8, np.log(lam0)], lo + 1e-12, hi - 1e-12)
        r = least_squares(res, x0, jac=jac, bounds=(lo, hi), method='trf', xtol=1e-15,
                          ftol=1e-15, gtol=1e-15, max_nfev=5000, x_scale='jac')
        if best is None or r.cost < best.cost: best = r
    w, lam = best.x[:M], np.exp(best.x[M:]); o = np.argsort(lam)
    return w[o], lam[o]

def Kf(w, lam, t): return (w[None, :] * np.exp(-np.outer(t, lam))).sum(1)

def errs(w, lam, w0, l0):
    Kt = Kf(w0, l0, kg); Kh = Kf(w, lam, kg)
    EK = np.sqrt(np.sum(kw * (Kh - Kt)**2) / np.sum(kw * Kt**2))
    p0 = np.r_[w0, l0]; ph = np.r_[w, lam]
    return EK, np.linalg.norm(ph - p0) / np.linalg.norm(p0)

def linpred(w0, l0, eps):
    _, _, J = Jmat(w0, l0); G = Gmat(w0, l0); b = Jmat(w0, l0)[0] @ w0
    brms = np.sqrt(np.mean(b**2)); Jp = np.linalg.pinv(J)
    Kn = np.sqrt(np.sum(kw * Kf(w0, l0, kg)**2))
    return (eps * brms * np.linalg.norm(G @ Jp, 'fro') / Kn,
            eps * brms * np.linalg.norm(Jp, 'fro') / np.linalg.norm(np.r_[w0, l0]))

if __name__ == "__main__":
    cases = {"benchmark (0.5,2,8)": (np.array([1, .6, .3]), np.array([.5, 2., 8.])),
             "r=4 (1,4,16)": (np.ones(3), np.array([1., 4., 16.]))}
    print("Table 3: first-order predictions vs nonlinear fits (percent)")
    for name, (w0, l0) in cases.items():
        b = Jmat(w0, l0)[0] @ w0; brms = np.sqrt(np.mean(b**2))
        for eps in [1e-6, 1e-5, 1e-4, 1e-3]:
            R, KR, KO = [], [], []
            for d in range(10):
                bb = b + eps * brms * np.random.default_rng(1000 + d).standard_normal(40)
                w, lam = fit(bb, 3, np.random.default_rng(5000 + d))
                R.append(errs(w, lam, w0, l0))
                kr, ko, _, _ = factor(Jmat(w, lam)[2], Gmat(w, lam)); KR.append(kr); KO.append(ko)
            R = 100 * np.array(R); EKl, Epl = linpred(w0, l0, eps)
            print(f"{name} eps={eps:.0e}: EK_lin={100*EKl:.3g} EK={R[:,0].mean():.3g}+-{R[:,0].std(ddof=1):.2g}"
                  f" | Ep_lin={100*Epl:.3g} Ep={R[:,1].mean():.3g}+-{R[:,1].std(ddof=1):.2g}"
                  f" | at estimates: krep median {np.median(KR):.3g} [{min(KR):.3g},{max(KR):.3g}],"
                  f" kobs median {np.median(KO):.3g} [{min(KO):.3g},{max(KO):.3g}]", flush=True)

    print("\nTable 4 (absolute-noise columns): excitations normalised to unit L2 norm on (0,1)")
    w0, l0 = cases["benchmark (0.5,2,8)"]; G = Gmat(w0, l0)
    Kn = np.sqrt(np.sum(kw * Kf(w0, l0, kg)**2))
    exc = [lambda x: 1 + x, lambda x: np.ones_like(x), lambda x: np.exp(-2 * x),
           lambda x: np.sin(2 * np.pi * x), lambda x: x**2]
    names = ["1+t", "1", "exp(-2t)", "sin(2 pi t)", "t^2"]
    def normed(f):
        c = np.sqrt(quad(lambda x: f(np.array([x]))[0]**2, 0, 1)[0]); return lambda x: f(x) / c
    for nm, f in zip(names, exc):
        kr, ko, smin, B = factor(Jmat(w0, l0, normed(f))[2], G)
        print(f"single {nm}: kobs={ko:.3g} sigma_min={smin:.3g} ||A+||_F/||K||={np.linalg.norm(np.linalg.pinv(B),'fro')/Kn:.3g}")
    for n in (1, 2, 4):
        tgn = (np.arange(1, 240 // n + 1) / (240 // n))**2
        J = np.vstack([Jmat(w0, l0, normed(exc[m]), tgn)[2] for m in range(n)])
        kr, ko, smin, B = factor(J, G)
        print(f"fixed budget {n}x{240//n}: kobs={ko:.3g} sigma_min={smin:.3g} ||A+||_F/||K||={np.linalg.norm(np.linalg.pinv(B),'fro')/Kn:.3g}")
