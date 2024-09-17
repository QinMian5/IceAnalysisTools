# Author: Mian Qin
# Date Created: 7/2/24
import numpy as np
from scipy.integrate import quad, cumulative_simpson
from scipy.special import sph_harm
import matplotlib.pyplot as plt


def gaussian(x, mu, sigma):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def generate_truncated_gaussian(mu, sigma, alpha_c):
    def f(x):
        return np.clip(gaussian(x, mu, sigma) - gaussian(mu - alpha_c, mu, sigma), 0, np.inf)

    result, error = quad(f, mu - alpha_c, mu + alpha_c)
    C = 1 / result

    def truncated_gaussian(x):
        return C * f(x)

    return truncated_gaussian


def generate_hv_f(mu, sigma, alpha_c):
    f = generate_truncated_gaussian(mu=mu, sigma=sigma, alpha_c=alpha_c)
    x = np.linspace(mu - alpha_c, mu + alpha_c, 1000, endpoint=True)
    y = cumulative_simpson(-f(x), x=x, initial=1)

    def hv_f(r: float | np.ndarray):
        return np.interp(r, x, y)

    return hv_f


def calc_pos_vec(src: np.ndarray, dst: np.ndarray, box: np.ndarray) -> np.ndarray:
    r = np.expand_dims(dst, 0) - np.expand_dims(src, 1)
    lt = r < -box / 2
    gt = r > box / 2
    r = r + lt * box - gt * box
    return r


def calc_hv(r: np.ndarray, mu=0.35, sigma=0.01, alpha_c=0.02) -> np.ndarray:
    r_s = np.linalg.norm(r, axis=2)
    hv_f = generate_hv_f(mu=mu, sigma=sigma, alpha_c=alpha_c)
    hv = hv_f(r_s)
    return hv


def calc_Y6m(r: np.ndarray, m: int) -> np.ndarray:
    x, y, z = r[:, :, 0], r[:, :, 1], r[:, :, 2]
    r = np.linalg.norm(r, axis=2)
    phi = np.arccos(z / r)
    theta = np.arctan2(y, x)
    Y6m = sph_harm(m, 6, theta, phi)
    return Y6m


def calc_Nnn(hv):
    Nnn = np.sum(hv, axis=1, keepdims=True)
    return Nnn


def calc_qbar6(r: np.ndarray, mu=0.35, sigma=0.01, alpha_c=0.02, restrict_ntilde_nn=True) -> np.ndarray:
    hv = calc_hv(r, mu, sigma, alpha_c)
    Nnn = calc_Nnn(hv)
    temp = 0
    for m in range(-6, 7):
        Y6m = calc_Y6m(r, m)
        q6m = 1 / Nnn * np.sum(hv * Y6m, axis=1, keepdims=True)
        qbar6m = 1 / (1 + Nnn) * (q6m + np.sum(hv * q6m.T, axis=1, keepdims=True))
        temp += np.abs(qbar6m) ** 2
    qbar6 = np.sqrt(4 * np.pi / 13 * temp)
    if restrict_ntilde_nn:
        qbar6[np.abs(Nnn - 4) > 0.5] = 0
    return qbar6


def classify(center_pos: np.ndarray, surrounding_pos: np.ndarray | None, box,
             mu=0.35, sigma=0.01, alpha_c=0.02, qbar_c=0.352):
    if surrounding_pos is None:
        pos = center_pos
    else:
        pos = np.concatenate((center_pos, surrounding_pos), axis=0)
    r = calc_pos_vec(pos, pos, box)
    M, N, _ = r.shape
    r = r + np.expand_dims(100 * np.eye(M, N), axis=2)
    qbar6 = calc_qbar6(r, mu, sigma, alpha_c).reshape(-1)
    return np.where(qbar6 > qbar_c)[0]


def main():
    hv_f = generate_hv_f(0.35, 0.01, 0.02)
    x = np.linspace(0.3, 0.4, 1000)
    y = hv_f(x)
    plt.plot(x, y)
    plt.savefig("test.png")


if __name__ == "__main__":
    main()
