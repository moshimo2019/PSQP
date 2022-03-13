import cvxpy as cp

from func import *


def normalize(d):
    n = d.shape[0]
    i, j = d.argsort(axis=1).argsort(1), d.argsort(axis=0).argsort(0)
    quartile = np.take_along_axis(d, i[:, int(n/4), None], axis=1)
    dq = d / quartile
    phi = i + j + 2
    ret = np.exp(-dq-phi)
    # Threshold 1e-6
    ret[ret < 1e-6] = 0
    return ret


def compute_dis(m, met=PM):
    mr = np.rot90(m, 1, [1, 2])
    lr, tb = met(m, m), met(mr, mr)
    max_ = max(lr.max(), tb.max()) * 100
    np.fill_diagonal(lr, max_), np.fill_diagonal(tb, max_)
    lr, tb = normalize(lr), normalize(tb)
    return lr, tb


def compute_grad(p, lr, tb):
    n = p.shape[0]
    index = np.arange(n, dtype=int).reshape(int(n ** 0.5), -1)
    lr_index = np.stack((index[:, :-1], index[:, 1:]), axis=-1).reshape(-1, 2)
    tb_index = np.stack((index[:-1], index[1:]), axis=-1).reshape(-1, 2)
    grad = np.zeros_like(p)
    for i, j in lr_index:
        grad[i] += lr @ p[j]
        grad[j] += p[i] @ lr
    for i, j in tb_index:
        grad[i] += tb @ p[j]
        grad[j] += p[i] @ tb
    target = (p @ lr @ p.T)[index[:, :-1], index[:, 1:]].sum() + \
        (p @ tb @ p.T)[index[:-1], index[1:]].sum()
    return target, grad


def PGD(p, active):
    n = p.shape[0]
    x = cp.Variable((n, n))
    constraint = [cp.sum(x[i]) == 1 for i in range(n)] + \
        [cp.sum(x.T[i]) == 1 for i in range(n)] + [x >= 0, x <= 1]
    for bound in [0, 1]:
        if (index := is_close(p, bound) & ~active).any():
            constraint.append(x[index] == bound)
    tar = cp.pnorm(x - p, p=2)
    prob = cp.Problem(cp.Minimize(tar), constraint)
    prob.solve()  # , verbose=True
    assert x.value is not None
    return x.value


def QP(m):
    n = m.shape[0]
    lr, tb = compute_dis(m)
    p = np.eye(n, dtype=float)
    active = np.ones(p.shape, dtype=bool)
    cnt, epoch = 0, 0
    p[active] = 1 / (n - cnt)
    while cnt < n:
        target, grad = compute_grad(p, lr, tb)
        epoch += 1
        print(f'step {epoch}({cnt}): {target:.7f}')
        grad[~active] = 0
        p += grad
        p = PGD(p, active)
        cnt += is_close(p[active], 1).sum()
        active[is_close(p, 0) | is_close(p, 1)] = False
        p[is_close(p, 0)], p[is_close(p, 1)] = 0, 1
    print(f'It totally takes {epoch} steps.')
    write(f'data/result/{S}.png', p @ m.reshape(n, -1))
    return


def main():
    m = read('data/lena256.bmp')
    write(f'data/permute/{S}.png', m[np.random.permutation(m.shape[0])])
    m = read(f'data/permute/{S}.png')
    QP(m)
    return


if __name__ == '__main__':
    main()
