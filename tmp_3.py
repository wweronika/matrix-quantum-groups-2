import sympy as sp
q, t, a, b, c, lam = sp.symbols("q t a b c lam")

q * t * (t-1) * b + (-t^2 + t - q**2)  * c = (lam + 1) * a
t * (t-1) * c + (-t^2 + t - q) * a = (lam + q**2) * b
q**2 * t * (t-1) * a + (-t^2 + t - 1) * b = (lam + q) * c