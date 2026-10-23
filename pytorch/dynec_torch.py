"""
DyNeC (PyTorch, GPU) -- Robust learning from noisy labels via class-conditional
consensus of training dynamics and latent neighbourhood agreement.

Supports CIFAR-10 / CIFAR-100 with synthetic noise (sym / asym) and the
real-world CIFAR-10N / CIFAR-100N labels (http://noisylabels.com, file
CIFAR-10_human.pt / CIFAR-100_human.pt).

Example:
  python dynec_torch.py --dataset cifar10 --noise sym --rate 0.5
  python dynec_torch.py --dataset cifar10 --cifarn CIFAR-10_human.pt --cifarn_key worse_label
  python dynec_torch.py --dataset cifar100 --cifarn CIFAR-100_human.pt --cifarn_key noisy_label
"""
import argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T
from sklearn.mixture import GaussianMixture

p = argparse.ArgumentParser()
p.add_argument('--dataset', default='cifar10'); p.add_argument('--noise', default='sym')
p.add_argument('--rate', type=float, default=0.5); p.add_argument('--cifarn', default='')
p.add_argument('--cifarn_key', default='worse_label'); p.add_argument('--epochs', type=int, default=150)
p.add_argument('--warm', type=int, default=10); p.add_argument('--k', type=int, default=10)
p.add_argument('--knn_every', type=int, default=3); p.add_argument('--lr', type=float, default=0.02)
p.add_argument('--bs', type=int, default=128); p.add_argument('--seed', type=int, default=0)
p.add_argument('--no_dyn', action='store_true'); p.add_argument('--no_nbr', action='store_true')
p.add_argument('--no_classcond', action='store_true'); p.add_argument('--no_relw', action='store_true')
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed); dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------- data ----------------
C = 10 if a.dataset == 'cifar10' else 100
DS = torchvision.datasets.CIFAR10 if C == 10 else torchvision.datasets.CIFAR100
mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
tr_tf = T.Compose([T.RandomCrop(32, 4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
te_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
train = DS('./data', True, download=True, transform=tr_tf)
train_eval = DS('./data', True, download=True, transform=te_tf)
test = DS('./data', False, download=True, transform=te_tf)
y_true = np.array(train.targets)
if a.cifarn:
    y_noisy = np.array(torch.load(a.cifarn)[a.cifarn_key])
else:
    rng = np.random.RandomState(a.seed); y_noisy = y_true.copy(); flip = rng.rand(len(y_true)) < a.rate
    if a.noise == 'sym':
        for i in np.where(flip)[0]: y_noisy[i] = rng.choice([c for c in range(C) if c != y_true[i]])
    else:  # standard CIFAR-10 asymmetric map; pair-flip for CIFAR-100
        amap = {9: 1, 2: 0, 3: 5, 5: 3, 4: 7} if C == 10 else {c: (c + 1) % C for c in range(C)}
        for i in np.where(flip)[0]: y_noisy[i] = amap.get(y_true[i], y_true[i])
print('actual noise rate %.3f' % (y_noisy != y_true).mean())
train.targets = y_noisy.tolist(); train_eval.targets = y_noisy.tolist()

class Idx(torch.utils.data.Dataset):
    def __init__(s, d): s.d = d
    def __len__(s): return len(s.d)
    def __getitem__(s, i): x, y = s.d[i]; return x, y, i
tl = torch.utils.data.DataLoader(Idx(train), a.bs, shuffle=True, num_workers=4, drop_last=True)
el = torch.utils.data.DataLoader(train_eval, 512, shuffle=False, num_workers=4)
sl = torch.utils.data.DataLoader(test, 512, shuffle=False, num_workers=4)

# ---------------- model: PreAct ResNet-18 ----------------
class PB(nn.Module):
    def __init__(s, i, o, st):
        super().__init__(); s.b1 = nn.BatchNorm2d(i); s.c1 = nn.Conv2d(i, o, 3, st, 1, bias=False)
        s.b2 = nn.BatchNorm2d(o); s.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False)
        s.sc = nn.Conv2d(i, o, 1, st, bias=False) if st != 1 or i != o else None
    def forward(s, x):
        o = F.relu(s.b1(x)); sc = s.sc(o) if s.sc is not None else x
        o = s.c1(o); o = s.c2(F.relu(s.b2(o))); return o + sc
class PreActResNet18(nn.Module):
    def __init__(s, C):
        super().__init__(); s.c = nn.Conv2d(3, 64, 3, 1, 1, bias=False); L = []; i = 64
        for o, st in [(64, 1), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1)]:
            L.append(PB(i, o, st)); i = o
        s.L = nn.Sequential(*L); s.bn = nn.BatchNorm2d(512); s.fc = nn.Linear(512, C)
    def forward(s, x, feat=False):
        h = F.adaptive_avg_pool2d(F.relu(s.bn(s.L(s.c(x)))), 1).flatten(1)
        return (s.fc(h), h) if feat else s.fc(h)
net = PreActResNet18(C).to(dev)
opt = torch.optim.SGD(net.parameters(), a.lr, momentum=0.9, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

# ---------------- DyNeC state ----------------
N = len(y_noisy); yn = torch.tensor(y_noisy, device=dev); Y = F.one_hot(yn, C).float()
msum = torch.zeros(N, device=dev); mcnt = 0
pbar = torch.full((N, C), 1. / C, device=dev); w = torch.ones(N, device=dev); Tgt = Y.clone(); nb = None

def gmm_clean(v, seed=0):
    v = v.reshape(-1, 1).astype(np.float64)
    if len(v) < 10 or v.std() < 1e-6: return np.ones(len(v))
    g = GaussianMixture(2, reg_covar=1e-3, max_iter=50, random_state=seed).fit(v)
    return g.predict_proba(v)[:, np.argmax(g.means_.ravel())]
def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))

@torch.no_grad()
def embed():
    net.eval(); E = []
    for x, _ in el: E.append(net(x.to(dev), feat=True)[1])
    net.train(); return F.normalize(torch.cat(E), dim=1)
@torch.no_grad()
def knn(E, k, bs=4096):
    out = []
    for i in range(0, len(E), bs):
        S = E[i:i + bs] @ E.T; S[torch.arange(S.shape[0]), torch.arange(i, i + S.shape[0])] = -1e9
        out.append(S.topk(k, dim=1).indices)
    return torch.cat(out)

@torch.no_grad()
def refine(ep):
    global w, Tgt, nb
    yc = y_noisy; n = N; mu = (msum / max(mcnt, 1)).cpu().numpy()
    if not a.no_nbr:
        if nb is None or (ep - a.warm) % a.knn_every == 0: nb = knn(embed(), a.k)
        wn = w[nb] if not a.no_relw else torch.ones_like(nb, dtype=torch.float)
        agree = ((wn * (yn[nb] == yn[:, None]).float()).sum(1) / (wn.sum(1) + 1e-8)).cpu().numpy()
        Q = (wn[..., None] * Y[nb]).sum(1) / (wn.sum(1, keepdim=True) + 1e-8)
    groups = [np.arange(n)] if a.no_classcond else [np.where(yc == c)[0] for c in range(C)]
    pd, pn, prior = np.ones(n), np.ones(n), np.full(n, .5)
    for g in groups:
        if not a.no_dyn: pd[g] = gmm_clean(mu[g], a.seed)
        if not a.no_nbr: pn[g] = gmm_clean(agree[g], a.seed)
        prior[g] = np.clip((pd[g] if not a.no_dyn else pn[g]).mean(), .05, .95)
    L = np.zeros(n)
    if not a.no_dyn: L += logit(pd)
    if not a.no_nbr: L += logit(pn)
    if not a.no_dyn and not a.no_nbr: L -= logit(prior)   # product-of-experts log-odds fusion
    w = torch.tensor(1 / (1 + np.exp(-L)), device=dev, dtype=torch.float)
    base = 0.5 * pbar + 0.5 * Q if not a.no_nbr else pbar
    sh = base ** 2; sh = sh / sh.sum(1, keepdim=True)          # sharpening, T = 0.5
    Tgt = w[:, None] * Y + (1 - w[:, None]) * sh
    clean = torch.tensor(y_noisy == y_true, device=dev)
    print('  sel-prec %.3f sel-rec %.3f corr-acc(noisy) %.3f' % (
        clean[w > .5].float().mean(), (w > .5)[clean].float().mean(),
        (Tgt.argmax(1)[~clean] == torch.tensor(y_true, device=dev)[~clean]).float().mean()))

@torch.no_grad()
def test_acc():
    net.eval(); c = 0
    for x, y in sl: c += (net(x.to(dev)).argmax(1).cpu() == y).sum().item()
    net.train(); return c / len(test)

for ep in range(a.epochs):
    if ep >= a.warm: refine(ep)
    for x, y, i in tl:
        x, i = x.to(dev), i.to(dev); z = net(x); pr = z.softmax(1)
        with torch.no_grad():
            pbar[i] = 0.7 * pbar[i] + 0.3 * pr
            py = pr.gather(1, yn[i][:, None]).squeeze(1)
            other = pr.scatter(1, yn[i][:, None], -1).max(1).values
            msum[i] += py - other
        loss = F.cross_entropy(z, yn[i]) if ep < a.warm else -(Tgt[i] * F.log_softmax(z, 1)).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    mcnt += 1; sched.step()
    print('epoch %d test acc %.4f' % (ep, test_acc()), flush=True)
