import numpy as np
import cv2


S = 32  # block size
B, SZ = [0, 0], [0, 0]


def read(filename, bs=S):
    image = cv2.imread(filename, 0)
    SZ[:] = image.shape
    B[:] = SZ[0] // bs, SZ[1] // bs
    m = image.reshape(B[0], bs, B[1], bs).swapaxes(1, 2).reshape(-1, bs, bs)
    return m.astype(float)


def write(filename, mat):
    image = mat.reshape(B[0], B[1], S, S).swapaxes(1, 2).reshape(SZ)
    cv2.imwrite(filename, image)


def is_close(a, b, atol=1e-5):
    return np.abs(a-b) < atol


def PM(m1, m2, p=1/10, q=3/16):
    sL = m1[..., -1] - m1[..., -2]
    sR = m2[..., 1] - m2[..., 0]
    m2, m1 = m2[None, ..., 0], m1[:, None, :, -1]
    d = np.abs(m2-m1-sL[:, None]) ** p + np.abs(m2-m1-sR[None, :]) ** p
    d = d.sum(-1) ** q/p
    return d


def MGC(mat1, mat2):
    uL = (mat1[..., -1] - mat1[..., -2]).mean(axis=1)
    uR = (mat2[..., 1] - mat2[..., 0]).mean(axis=1)
    GLR = mat2[None, ..., 0] - mat1[:, None, :, -1]
    dLR = ((GLR - uL[:, None, None]) ** 2).mean(axis=2)
    dRL = ((GLR - uR[None, :, None]) ** 2).mean(axis=2)
    cLR = dLR + dRL
    return cLR
