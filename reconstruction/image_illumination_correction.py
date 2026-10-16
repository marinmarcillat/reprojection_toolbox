import cv2
import numpy as np
from collections import deque
from pathlib import Path
from tqdm import tqdm

# Python port of Matisse Preprocessing (Arnaubec, 2023)


GAMMA = 0.2

def gamma_undo(v):   # display -> linear
    return np.power(np.clip(v, 0, None), GAMMA)

def gamma_do(v):     # linear -> display
    return np.power(np.clip(v, 0, None), 1.0 / GAMMA)


def poly_terms(x, y):
    # [x^3, y^3, x*y^2, x^2*y, x^2, y^2, x*y, x, y, 1]
    return np.stack([x**3, y**3, x * y**2, x**2 * y, x**2, y**2,
                     x * y, x, y, np.ones_like(x)], axis=1)


def compensate_channel(img, img_low, med_img, mask_low, scaling, rng,
                       out_perc=0.01, max_corr=10.0):
    """img: uint8 full (prepro) res; img_low: uint8 low-res of the ORIGINAL image;
    med_img: float32 linear temporal median (low-res);
    mask_low: uint8 low-res mask (nonzero = invalid) or None;
    scaling = lowres_scale / prepro_scale."""
    med = cv2.medianBlur(med_img.astype(np.float32), 5)

    lo, hi = np.quantile(med, [out_perc, 1 - out_perc])
    invalid = (med <= lo) | (med >= hi)
    if mask_low is not None:
        invalid |= cv2.medianBlur(mask_low, 5) > 0

    ys, xs = np.nonzero(~invalid)
    if len(xs) > 7500:
        idx = rng.choice(len(xs), 5000, replace=False)
        ys, xs = ys[idx], xs[idx]
    z = med[ys, xs].astype(np.float64)

    alpha, *_ = np.linalg.lstsq(poly_terms(xs.astype(np.float64),
                                           ys.astype(np.float64)), z, rcond=None)

    # evaluate the surface on the full-res grid (broadcasting, no huge design matrix)
    h, w = img.shape
    X = (np.arange(w) * scaling)[None, :]
    Y = (np.arange(h) * scaling)[:, None]
    a = alpha
    zz = (a[0]*X**3 + a[1]*Y**3 + a[2]*X*Y**2 + a[3]*X**2*Y + a[4]*X**2
          + a[5]*Y**2 + a[6]*X*Y + a[7]*X + a[8]*Y + a[9])
    with np.errstate(divide="ignore"):
        corr = 1.0 / zz
    corr = np.where((corr > max_corr) | (zz < 0), max_corr, corr)

    out = corr * gamma_undo(img / 255.0)

    # exposure matching on the low-res version
    out_low = cv2.resize(out, (img_low.shape[1], img_low.shape[0]))  # bilinear
    q_in = np.quantile(gamma_undo(img_low / 255.0), 0.8)
    q_c80, q_c999 = np.quantile(out_low, [0.8, 0.999])
    scale = q_in / q_c80
    if scale * q_c999 > 1.0:
        scale = 1.0 / q_c999

    return np.clip(np.rint(255.0 * gamma_do(scale * out)), 0, 255).astype(np.uint8)


def preprocess_sequence(files, out_dir, ws=21, prepro_scale=1.0, mask=None, seed=0):
    ws += (ws % 2 == 0)                      # force odd, as in the C++
    n, half = len(files), (ws - 1) // 2
    flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    if n < ws:
        raise ValueError("fewer images than the window size: C++ disables compensation")

    first = cv2.imread(str(files[0]), flags)
    low_s = min(1.0, np.sqrt(1e5 / (first.shape[0] * first.shape[1])))
    mask_low = None
    if mask is not None:
        mask_low = cv2.resize(mask, None, fx=low_s, fy=low_s) if low_s < 1 else mask

    def load_low(f):
        im = cv2.imread(str(f), flags)
        return im, cv2.resize(im, None, fx=low_s, fy=low_s)

    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    start = 0
    window = deque(load_low(files[j])[1] for j in range(ws))   # low-res BGR frames

    for i, f in tqdm(enumerate(files), desc="Illumination correction", total=n, unit="img"):
        # same window logic as the C++ pop_front/push_back: centred, clamped at the ends
        target = min(max(i - half, 0), n - ws)
        while start < target:
            window.popleft()
            window.append(load_low(files[start + ws])[1])
            start += 1

        img, img_low = load_low(f)
        if prepro_scale < 1.0:
            img = cv2.resize(img, None, fx=prepro_scale, fy=prepro_scale)

        stack = gamma_undo(np.stack(list(window)) / 255.0)     # (ws, H, W, 3), linear
        median = np.median(stack, axis=0).astype(np.float32)    # temporal median

        chans = [compensate_channel(img[..., c], img_low[..., c], median[..., c],
                                    mask_low, low_s / prepro_scale, rng)
                 for c in range(3)]
        cv2.imwrite(str(out_dir / Path(f).name), cv2.merge(chans))