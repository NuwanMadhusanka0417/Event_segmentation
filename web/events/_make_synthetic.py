"""Tiny seed file so the UI has something to play with."""
from pathlib import Path
import numpy as np
from scipy.io import savemat

rng = np.random.default_rng(0)
N = 4000
t = np.sort(rng.uniform(0, 1.0, N))
x = (10 + (t * 110) + rng.normal(0, 4, N)).clip(0, 127)
y = (40 + 30 * np.sin(t * 6) + rng.normal(0, 3, N)).clip(0, 95)
p = rng.choice([0, 1], N)
out = Path(__file__).with_name("synthetic.mat")
savemat(str(out), {"t": t, "x": x, "y": y, "p": p})
print(f"wrote {out} with {N} events")
