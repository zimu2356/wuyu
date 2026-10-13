"""Independent batched Elman networks, explicit four-step BPTT in NumPy."""
import numpy as np


class RNN:
    def __init__(self, seeds, inputs=17, hidden=24, memory=True):
        self.n, self.d, self.k = len(seeds), inputs, hidden
        self.memory = memory
        self.w = np.empty((self.n, inputs, hidden))
        self.r = np.empty((self.n, hidden, hidden))
        self.v = np.empty((self.n, hidden))
        for i, seed in enumerate(seeds):
            rng = np.random.default_rng(int(seed)+900000)
            self.w[i] = rng.normal(0, 1/np.sqrt(inputs), (inputs, hidden))
            self.r[i] = .9*np.linalg.qr(rng.normal(size=(hidden, hidden)))[0]
            self.v[i] = rng.normal(0, .1, hidden)
        self.b = np.zeros((self.n, hidden))
        self.c = np.zeros(self.n)
        self.h = np.zeros((self.n, hidden))
        self.updates = np.zeros(self.n, dtype=np.int64)

    @property
    def parameter_count(self):
        return self.d*self.k+self.k*self.k+2*self.k+1

    def forward(self, x, h0):
        h = h0.copy()
        states, old = [], []
        for xt in x:
            if not self.memory:
                h = np.zeros_like(h)
            old.append(h)
            h = np.tanh(np.einsum('nd,ndh->nh', xt, self.w)+
                        np.einsum('ni,nih->nh', h, self.r)+self.b)
            states.append(h)
        logits = np.einsum('nh,nh->n', h, self.v)+self.c
        p = 1/(1+np.exp(-np.clip(logits, -35, 35)))
        return p, np.asarray(states), np.asarray(old)

    def gradients(self, x, h0, y, valid):
        p, hs, old = self.forward(x, h0)
        error = (p-y)*valid
        g = {'v':error[:, None]*hs[-1], 'c':error.copy(),
             'w':np.zeros_like(self.w), 'r':np.zeros_like(self.r), 'b':np.zeros_like(self.b)}
        dh = error[:, None]*self.v
        for t in range(len(x)-1, -1, -1):
            dz = dh*(1-hs[t]**2)
            g['w'] += x[t, :, :, None]*dz[:, None, :]
            g['r'] += old[t, :, :, None]*dz[:, None, :]
            g['b'] += dz
            dh = np.einsum('nh,nih->ni', dz, self.r) if self.memory else np.zeros_like(dz)
        return g

    def learn(self, x, h0, y, valid, rate):
        g = self.gradients(x, h0, y, valid)
        norm = np.sqrt(sum((a.reshape(self.n, -1)**2).sum(axis=1) for a in g.values()))
        scale = np.minimum(1., 5/np.maximum(norm, 1e-12))
        for name, a in g.items():
            a *= scale.reshape((self.n,)+(1,)*(a.ndim-1))
            setattr(self, name, getattr(self, name)-rate*a)
        self.updates += valid.astype(np.int64)

    def observe(self, x):
        p, h, _ = self.forward(x, self.h)
        self.h = h[-1].copy()
        return p


class Memory:
    """Actual feedback-labelled events; no balanced sampling or future labels."""
    def __init__(self, n, capacity=256, inputs=17, hidden=24):
        self.capacity = capacity
        self.x = np.zeros((n, capacity, 4, inputs))
        self.h = np.zeros((n, capacity, hidden))
        self.y = np.zeros((n, capacity))
        self.count = np.zeros(n, dtype=np.int64)

    def add(self, x, h, y, valid):
        rows = np.flatnonzero(valid)
        slots = self.count[rows] % self.capacity
        self.x[rows, slots] = x[:, rows].transpose(1, 0, 2)
        self.h[rows, slots] = h[rows]
        self.y[rows, slots] = y[rows]
        self.count[rows] += 1

    def sample(self, rng):
        n = len(self.count)
        slots = (rng.random(n)*np.maximum(1, np.minimum(self.count, self.capacity))).astype(int)
        rows = np.arange(n)
        return (self.x[rows, slots].transpose(1, 0, 2), self.h[rows, slots],
                self.y[rows, slots], self.count > 0)
