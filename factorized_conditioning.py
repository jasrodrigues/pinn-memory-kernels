"""
Factorized conditioning of memory-kernel identification.
Reproduces Tables 1-5 of the proceedings paper (training-free, CPU only).

J = A G, where G = dK/dp (kernel Jacobian, values on a weighted kernel grid)
and A is the Volterra observation operator K -> int_0^t K(t-s) u_t(s) ds.
kappa_rep = cond(G D_G)       (D_G: column normalisation of G)
kappa_obs = cond(A restricted to range(G)), computed as cond(J D_G V S^-1)
where G D_G = Q S V^T.  All Prony / stretched-exponential computations use
40-digit arithmetic (mpmath) to avoid cancellation; the spline uses doubles.
"""
import mpmath as mp
import numpy as np
from scipy.integrate import quad
mp.mp.dps = 40

# ---------- quadrature and grids (as in Rodrigues, Mathematics 2026) -------
def gauss_legendre01(n):
    x0, _ = np.polynomial.legendre.leggauss(n)
    X, W = [], []
    for k in range(n):
        r = mp.findroot(lambda z: mp.legendre(n, z), mp.mpf(x0[k]))
        dp = mp.diff(lambda z: mp.legendre(n, z), r)
        X.append((r + 1) / 2); W.append(1 / ((1 - r**2) * dp**2))
    return X, W
S, WS = gauss_legendre01(50)

def data_grid(T=1, Nt=40):
    return [T * mp.mpf(i)**2 / Nt**2 for i in range(1, Nt + 1)]

def kernel_grid(T=1, Ne=200):
    t = [T * mp.mpf(i)**2 / Ne**2 for i in range(1, Ne + 1)]
    w = []
    for i in range(Ne):
        a = t[i-1] if i > 0 else t[0]; b = t[i+1] if i < Ne-1 else t[-1]
        w.append((b - a) / 2)
    return t, w

# ---------- linear algebra helpers ----------
def svals(A): return mp.svd_r(A, compute_uv=False)
def cond(A):
    s = svals(A); return max(s) / min(s)
def colnorm(J):
    D = mp.matrix(J.cols, J.cols)
    for j in range(J.cols): D[j, j] = 1 / mp.norm(J[:, j])
    return D
def fro(A): return mp.sqrt(sum(A[i, j]**2 for i in range(A.rows) for j in range(A.cols)))
def stack(mats):
    rows = []
    for J in mats:
        rows += [[J[i, j] for j in range(J.cols)] for i in range(J.rows)]
    return mp.matrix(rows)

def factorize(J, G):
    """returns kappa(J D_J), kappa(J D_G), kappa_rep, kappa_obs"""
    DG = colnorm(G)
    U, Sg, V = mp.svd_r(G * DG)
    Si = mp.matrix(G.cols, G.cols)
    for k in range(G.cols): Si[k, k] = 1 / Sg[k]
    B = J * DG * V.T * Si
    return cond(J * colnorm(J)), cond(J * DG), max(Sg) / min(Sg), cond(B)

# ---------- generic kernel family ----------
def conv_rows(func, tg, gp):
    """rows of int_0^t func(t-s) gp(s) ds, func returns a list (one entry per column)"""
    rows = []
    for t in tg:
        acc = None
        for s, om in zip(S, WS):
            v = func(t * (1 - s)); g = gp(s * t)
            acc = [om * x * g for x in v] if acc is None else [a + om * x * g for a, x in zip(acc, v)]
        rows.append([t * a for a in acc])
    return mp.matrix(rows)

def kernel_rows(func, kg, kw):
    return mp.matrix([[mp.sqrt(om) * x for x in func(t)] for t, om in zip(kg, kw)])

def prony(lam, w):
    def dK(t):
        e = [mp.e**(-l * t) for l in lam]
        return e + [-wj * t * ej for wj, ej in zip(w, e)]
    def K(t): return [sum(wj * mp.e**(-l * t) for wj, l in zip(w, lam))]
    return dK, K

def stretched(A0, tau, beta):
    def dK(t):
        if t == 0: return [mp.mpf(1), mp.mpf(0), mp.mpf(0)]
        z = (t / tau)**beta; K = A0 * mp.e**(-z)
        return [K / A0, K * beta * z / tau, -K * z * mp.log(t / tau)]
    def K(t): return [A0 * mp.e**(-(t / tau)**beta)]
    return dK, K

def linear_prediction(J, G, Krows, b, p, eps=mp.mpf('1e-3')):
    """RMS first-order relative errors for b_i + eps*||b||_rms*zeta_i"""
    brms = mp.sqrt(sum(x**2 for x in b) / len(b))
    Jp = mp.inverse(J.T * J) * J.T
    Kn = mp.norm(Krows)
    return eps * brms * fro(G * Jp) / Kn, eps * brms * fro(Jp) / mp.norm(p)

g1 = lambda t: 1 + t
g2 = lambda t: mp.pi * mp.cos(2 * mp.pi * t)
EXC = [lambda t: 1 + t, lambda t: mp.mpf(1), lambda t: mp.e**(-2 * t),
       lambda t: mp.sin(2 * mp.pi * t), lambda t: t**2]
f = lambda x: mp.nstr(x, 3)

if __name__ == "__main__":
    tg = data_grid(); kg, kw = kernel_grid()
    w0 = [mp.mpf(1), mp.mpf('0.6'), mp.mpf('0.3')]

    print("Table 1: factorization at the true parameters")
    cases = [("Prony close", prony([mp.mpf('0.5'), 2, 8], w0)),
             ("Prony separated", prony([mp.mpf('0.1'), 5, 50], w0)),
             ("Stretched exp", stretched(mp.mpf('1.9'), mp.mpf('0.55'), mp.mpf('0.7')))]
    for name, (dK, K) in cases:
        G = kernel_rows(dK, kg, kw)
        for gn, gp in (("g1", g1), ("g2", g2)):
            print(name, gn, [f(x) for x in factorize(conv_rows(dK, tg, gp), G)])

    print("\nTable 2: separation sweep lambda=(1,r,r^2), w=(1,1,1)")
    for rs in ['1.2', '1.5', '2', '3', '4', '5', '7', '10']:
        r = mp.mpf(rs); lam = [mp.mpf(1), r, r**2]; w = [mp.mpf(1)] * 3
        dK, K = prony(lam, w)
        G = kernel_rows(dK, kg, kw); J = conv_rows(dK, tg, g1)
        b = [x[0] for x in conv_rows(K, tg, g1).tolist()]
        EK, Ep = linear_prediction(J, G, kernel_rows(K, kg, kw), b, mp.matrix(w + lam))
        k = factorize(J, G)
        print(rs, f(k[0]), f(k[2]), f(k[3]), "EK_lin=%s%% Ep_lin=%s%%" % (f(100*EK), f(100*Ep)))

    lam = [mp.mpf('0.5'), mp.mpf(2), mp.mpf(8)]
    dK, K = prony(lam, w0); G = kernel_rows(dK, kg, kw); Kr = kernel_rows(K, kg, kw)
    p0 = mp.matrix(w0 + lam)
    print("\nTable 3a: single excitations")
    for m, gp in enumerate(EXC):
        k = factorize(conv_rows(dK, tg, gp), G); print(m + 1, f(k[0]), f(k[3]))
    print("Table 3b: cumulative, 40 rows per trajectory")
    Js, bs = [], []
    for m, gp in enumerate(EXC):
        Js.append(conv_rows(dK, tg, gp)); bs += [x[0] for x in conv_rows(K, tg, gp).tolist()]
        if m + 1 in (1, 2, 3, 5):
            J = stack(Js); k = factorize(J, G); EK, Ep = linear_prediction(J, G, Kr, bs, p0)
            print(m + 1, f(k[0]), f(k[3]), "EK_lin=%s%%" % f(100*EK))
    print("Table 3c: fixed budget of 240 rows")
    for n in (1, 2, 4):
        tgn = data_grid(1, 240 // n)
        Js = [conv_rows(dK, tgn, EXC[m]) for m in range(n)]
        bs = sum([[x[0] for x in conv_rows(K, tgn, EXC[m]).tolist()] for m in range(n)], [])
        J = stack(Js); k = factorize(J, G); EK, Ep = linear_prediction(J, G, Kr, bs, p0)
        print("%dx%d" % (n, 240 // n), f(k[0]), f(k[3]), "EK_lin=%s%%" % f(100*EK))

    print("\nTable 4: observation window, fixed resolution (Nt=40T, Ne=200T)")
    mp.mp.dps = 30
    for T in (1, 2, 4, 8):
        tgT = data_grid(T, 40 * T); kgT, kwT = kernel_grid(T, 200 * T)
        GT = kernel_rows(dK, kgT, kwT); J = conv_rows(dK, tgT, g1)
        b = [x[0] for x in conv_rows(K, tgT, g1).tolist()]
        EK, Ep = linear_prediction(J, GT, kernel_rows(K, kgT, kwT), b, p0)
        k = factorize(J, GT)
        print(T, f(k[0]), f(k[2]), f(k[3]), "Ep_lin=%s%% EK_lin=%s%%" % (f(100*Ep), f(100*EK)))

    print("\nTable 5: piecewise-linear spline, kappa_obs and EK_lin (double precision)")
    xg, wg = np.polynomial.legendre.leggauss(3)
    def hat(j, kn, x):
        y = np.zeros_like(x, dtype=float); n = len(kn)
        if j > 0:
            m = (x >= kn[j-1]) & (x <= kn[j]); y[m] = (x[m] - kn[j-1]) / (kn[j] - kn[j-1])
        if j < n - 1:
            m = (x >= kn[j]) & (x <= kn[j+1]); y[m] = (kn[j+1] - x[m]) / (kn[j+1] - kn[j])
        return y
    def conv(j, kn, t):
        br = np.unique(np.concatenate([[0, t], kn[(kn > 0) & (kn < t)]])); s = 0.
        for a, b_ in zip(br[:-1], br[1:]):
            x = (a + b_) / 2 + (b_ - a) / 2 * xg
            s += np.sum(wg * (b_ - a) / 2 * hat(j, kn, x) * (1 + t - x))
        return s
    Ktrue = lambda t: np.exp(-0.5*t) + 0.6*np.exp(-2*t) + 0.3*np.exp(-8*t)
    tgd = (np.arange(1, 41) / 40)**2; kgd = (np.arange(1, 201) / 200)**2
    kwd = np.empty(200); kwd[1:-1] = (kgd[2:] - kgd[:-2]) / 2
    kwd[0] = (kgd[1] - kgd[0]) / 2; kwd[-1] = (kgd[-1] - kgd[-2]) / 2
    bd = np.array([quad(lambda s: Ktrue(t - s) * (1 + s), 0, t, epsabs=1e-13)[0] for t in tgd])
    for n in (8, 15, 25):
        for nm, kn in (("uniform", np.linspace(0, 1, n)), ("quadratic", np.linspace(0, 1, n)**2)):
            J = np.array([[conv(j, kn, t) for j in range(n)] for t in tgd])
            Gd = np.sqrt(kwd)[:, None] * np.array([hat(j, kn, kgd) for j in range(n)]).T
            U, Sg, Vt = np.linalg.svd(Gd, full_matrices=False)
            B = J @ Vt.T / Sg; sb = np.linalg.svd(B, compute_uv=False)
            EK = 1e-3 * np.sqrt(np.mean(bd**2)) * np.linalg.norm(np.linalg.pinv(B), 'fro') \
                 / np.sqrt(np.sum(kwd * Ktrue(kgd)**2))
            print(n, nm, "kappa_obs=%.3g" % (sb[0] / sb[-1]), "EK_lin=%.2f%%" % (100 * EK))
