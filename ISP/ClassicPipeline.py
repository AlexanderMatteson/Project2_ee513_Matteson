# imports:
import numpy as np
import rawpy
import matplotlib.pyplot as plt
import glob
import os
from scipy.ndimage import median_filter
import cv2
from scipy.signal import convolve2d
from scipy.ndimage import shift as nd_shift
from scipy.ndimage import uniform_filter
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio
import colour
from scipy.ndimage import convolve
# parameters:
RGGB_OFFSETS = {"R": (0, 0), "G1": (0, 1), "G2": (1, 0), "B": (1, 1)}
PLANE_COLORS = {"R": "tab:red", "G1": "tab:green", "G2": "olive", "B": "tab:blue"}
N_COLS, N_ROWS = 6, 4

TOL = 4 * np.finfo(np.float64).eps        # a few rounding steps
CENTER_FRAC = 0.04
PROFILE_BAND = 20  
GRID_COLS, GRID_ROWS = 6, 4        # 24-patch chart: 6 columns x 4 rows
CORE = 0.40                        # sample the central 40% (never more than half) of each patch
cc = colour.CCS_COLOURCHECKERS[
    "ColorChecker24 - After November 2014"
]
CH = {"R": 0, "G": 1, "B": 2}
PATCH_NAMES = [
    "dark skin", "light skin", "blue sky", "foliage",
    "blue flower", "bluish green", "orange", "purplish blue",
    "moderate red", "purple", "yellow green", "orange yellow",
    "blue", "green", "red", "yellow", "magenta", "cyan",
    "white", "neutral 8", "neutral 6.5", "neutral 5",
    "neutral 3.5", "black"
]


def channel_planes(raw):
    """Return ({name: 2D array}, {name: manufacturer black}) for each CFA channel."""
    img = raw.raw_image_visible.astype(np.float64)
    pat = raw.raw_pattern
    desc = raw.color_desc.decode()
    bl = raw.black_level_per_channel
    planes, mfr = {}, {}
    g_seen = 0
    for y in range(2):
        for x in range(2):
            idx = pat[y, x]
            name = desc[idx]
            if name == "G":
                g_seen += 1
                name = f"G{g_seen}"
            planes[name] = img[y::2, x::2]
            mfr[name] = bl[idx]
    return planes, mfr

def load_dng_mosaic(path):
    import rawpy
    with rawpy.imread(path) as raw:
        mosaic = raw.raw_image_visible.copy()
        desc = raw.color_desc.decode()
        idx = raw.raw_pattern.flatten()                 # color index at each 2x2 position
        pattern = "".join(desc[i] for i in idx)
        bl = np.array(raw.black_level_per_channel, dtype=np.float64)
        black = bl[idx]                                 # reorder to 2x2 raster order
        white = float(raw.white_level)
    return {"mosaic": mosaic, "pattern": pattern, "black": black, "white": white}

# ----------------------------------------------------------------------------
# 2. Pattern check
# ----------------------------------------------------------------------------
def check_rggb(pattern):
    ok = pattern == "RGGB"
    print(f"CFA pattern: {pattern}  ->  {'OK (RGGB)' if ok else 'NOT RGGB'}")
    if not ok:
        raise ValueError(
            f"Expected RGGB but got {pattern}. Crop/shift the mosaic or remap offsets "
            "before continuing (an odd crop offset changes the pattern)."
        )
    return ok


def split_planes(mosaic):
    return {k: mosaic[r::2, c::2] for k, (r, c) in RGGB_OFFSETS.items()}


# ----------------------------------------------------------------------------
# 3. Clipping check
# ----------------------------------------------------------------------------
def clipping_report(mosaic, white, tol=0):
    """Fraction of pixels at/near saturation, plus max & percentile per channel."""
    print(f"White level: {white:.0f}   (data max overall: {mosaic.max()})")
    out = {}
    for k, p in split_planes(mosaic).items():
        frac = float(np.mean(p >= white - tol))
        out[k] = dict(max=int(p.max()), p99_9=float(np.percentile(p, 99.9)),
                      frac_clipped=frac)
        flag = "  <-- CLIPPING" if frac > 1e-4 else ""
        print(f"  {k:>2}: max={p.max():6d}  p99.9={out[k]['p99_9']:8.1f}  "
              f"clipped={100*frac:.4f}%{flag}")
    return out


# ----------------------------------------------------------------------------
# 4. Horizontal profile plot
# ----------------------------------------------------------------------------
def plot_horizontal_profiles(mosaic, black, save_path=None, title="Flat field horizontal profile"):
    import matplotlib.pyplot as plt

    planes = split_planes(mosaic.astype(np.float64))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, (k, p) in enumerate(planes.items()):
        row = p[p.shape[0] // 2, :] - black[i]
        x = np.arange(row.size) * 2
        axes[0].plot(x, row, color=PLANE_COLORS[k], label=k, lw=0.9)
        axes[1].plot(x, row / np.median(row[row.size // 2 - 25: row.size // 2 + 25]),
                     color=PLANE_COLORS[k], label=k, lw=0.9)
    axes[0].set(title="Black-subtracted (DN)", xlabel="x (mosaic pixel)", ylabel="DN")
    axes[1].set(title="Normalized to center", xlabel="x (mosaic pixel)", ylabel="relative")
    for a in axes:
        a.grid(alpha=0.3); a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ----------------------------------------------------------------------------
# 5. Gain map from flat field
# ----------------------------------------------------------------------------
def build_gain_map(flat_mosaic, black, center_frac=0.05, smooth_sigma=8.0,
                   max_gain=8.0):
    """
    Per-plane (R, G1, G2, B) gain map, same shape as the mosaic.

    For each plane:  flat = plane - black   (linear light, black removed)
                     flat = optional gaussian smoothing (suppresses noise)
                     gain = center / flat,  center = median of central patch
    Returns float32 array HxW (gain at each mosaic site).
    """
    H, W = flat_mosaic.shape
    gain = np.ones((H, W), dtype=np.float32)
    planes = split_planes(flat_mosaic.astype(np.float64))

    for i, (k, (r, c)) in enumerate(RGGB_OFFSETS.items()):
        p = planes[k] - black[i]
        p = np.clip(p, 1.0, None)                      # avoid divide by <=0
        if smooth_sigma and smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter
            p = gaussian_filter(p, smooth_sigma, mode="nearest")
        h, w = p.shape
        ch, cw = max(int(h * center_frac), 4), max(int(w * center_frac), 4)
        center = np.median(p[h // 2 - ch: h // 2 + ch, w // 2 - cw: w // 2 + cw])
        g = np.clip(center / p, None, max_gain)
        gain[r::2, c::2] = g.astype(np.float32)
    return gain


# ----------------------------------------------------------------------------
# 6. Apply
# ----------------------------------------------------------------------------
def apply_lens_shading(mosaic, gain, black, white=None, keep_black_level=False):
    """
    corrected = (mosaic - black) * gain   (float32, linear, black-subtracted)
    If keep_black_level, black is added back so downstream steps see raw-like DN.
    """
    if gain.shape != mosaic.shape:
        import cv2
        gain = cv2.resize(gain, (mosaic.shape[1], mosaic.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
    out = np.empty(mosaic.shape, dtype=np.float32)
    m = mosaic.astype(np.float32)
    for i, (k, (r, c)) in enumerate(RGGB_OFFSETS.items()):
        out[r::2, c::2] = np.clip(m[r::2, c::2] - black[i], 0, None)
    out *= gain
    if white is not None:
        out = np.clip(out, 0, white - float(np.mean(black)))
    if keep_black_level:
        for i, (k, (r, c)) in enumerate(RGGB_OFFSETS.items()):
            out[r::2, c::2] += black[i]
    return out


# ----------------------------------------------------------------------------
# Reusable entry point: calibrate once, correct many images
# ----------------------------------------------------------------------------
def calibrate_from_flat(flat_dng_path, gain_out="lsc_gain.npz", **kw):
    d = load_dng_mosaic(flat_dng_path)
    check_rggb(d["pattern"])
    np.savez(gain_out, gain=build_gain_map(d["mosaic"], d["black"], **kw),
             black=d["black"], white=d["white"])
    return gain_out


def correct_lens_shading(image, gain_npz="lsc_gain.npz", keep_black_level=False):
    """
    image: path to a DNG, OR a dict from load_dng_mosaic, OR a raw mosaic ndarray.
    Returns float32 linear mosaic (black-subtracted unless keep_black_level).
    """
    cal = np.load(gain_npz)
    gain, cal_black, white = cal["gain"], cal["black"], float(cal["white"])
    if isinstance(image, str):
        image = load_dng_mosaic(image)
    if isinstance(image, dict):
        check_rggb(image["pattern"])
        mosaic, black = image["mosaic"], image["black"]
    else:
        mosaic, black = image, cal_black      # bare array: assume same sensor settings
    return apply_lens_shading(mosaic, gain, black, white, keep_black_level)



def split_planes(mosaic):
    return {k: mosaic[r::2, c::2] for k, (r, c) in RGGB_OFFSETS.items()}


def _robust_sigma(x):
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


# ----------------------------------------------------------------------------
# Loading dark frames
# ----------------------------------------------------------------------------
def load_dark_stack(folder):
    """Average every DNG in `folder`. Returns dict: mean, std, black, white, n."""
    paths = sorted(p for p in glob.glob(os.path.join(folder, "*"))
                   if p.lower().endswith(".dng"))
    if not paths:
        raise FileNotFoundError(f"No .dng files in {folder}")
    s = s2 = None
    for p in paths:
        d = load_dng_mosaic(p)
        if d["pattern"] != "RGGB":
            raise ValueError(f"{p}: pattern {d['pattern']} is not RGGB")
        m = d["mosaic"].astype(np.float64)
        if s is None:
            s, s2, shape = np.zeros_like(m), np.zeros_like(m), m.shape
            black, white = d["black"], d["white"]
        elif m.shape != shape:
            raise ValueError(f"{p}: shape {m.shape} differs from {shape}")
        s += m
        s2 += m * m
    n = len(paths)
    mean = s / n
    std = np.sqrt(np.clip(s2 / n - mean ** 2, 0, None))
    print(f"Loaded {n} dark frames, shape {shape}, black {black}")
    return {"mean": mean, "std": std, "black": black, "white": white, "n": n}


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------
def detect_hot_pixels(master_dark, black, k=8.0, min_excess=3.0):
    """
    Hot / stuck-high pixels from the averaged dark frame.
    Per CFA plane: residual = (dark - black) - 3x3 same-colour median.
    Flag residual > max(k * robust_sigma, min_excess DN).
    """
    mask = np.zeros(master_dark.shape, dtype=bool)
    for i, (name, (r, c)) in enumerate(RGGB_OFFSETS.items()):
        p = master_dark[r::2, c::2] - black[i]
        resid = p - median_filter(p, size=3, mode="nearest")
        thr = max(k * max(_robust_sigma(resid), 0.1), min_excess)
        mask[r::2, c::2] = resid > thr
        print(f"  hot  {name:>2}: thr={thr:6.2f} DN  flagged={mask[r::2, c::2].sum()}")
    return mask


def detect_flat_defects(flat_mosaic, black, k=6.0, min_dev=0.10):
    """
    Dead / stuck-low / off-gain pixels from one flat.
    Per plane: resid = pixel - 3x3 same-colour median (the smooth vignette cancels).
    Noise is larger (relatively) in the dim corners, so the threshold uses a LOCAL
    noise estimate (mean |resid| over a 65x65 window), not one global sigma.
    Flag |resid| > max(k * local_sigma, min_dev * local_median).
    """
    from scipy.ndimage import uniform_filter
    mask = np.zeros(flat_mosaic.shape, dtype=bool)
    for i, (name, (r, c)) in enumerate(RGGB_OFFSETS.items()):
        p = flat_mosaic[r::2, c::2].astype(np.float64) - black[i]
        med = median_filter(p, size=3, mode="nearest")
        resid = p - med
        local_sigma = uniform_filter(np.abs(resid), 65, mode="nearest") / 0.7979
        thr = np.maximum(k * local_sigma, min_dev * np.clip(med, 1.0, None))
        mask[r::2, c::2] = np.abs(resid) > thr
        print(f"  flat {name:>2}: median local sigma={np.median(local_sigma):.2f} DN  "
              f"flagged={mask[r::2, c::2].sum()}")
    return mask


def build_defect_map(dark_folder, flat_path, out_path="defect_map.npz",
                     hot_k=8.0, hot_min_excess=3.0, flat_k=6.0, flat_min_dev=0.10):
    """Build and save the boolean defect map (True = defective)."""
    dark = load_dark_stack(dark_folder)
    flat = load_dng_mosaic(flat_path)
    if flat["pattern"] != "RGGB":
        raise ValueError(f"Flat pattern {flat['pattern']} is not RGGB")
    if flat["mosaic"].shape != dark["mean"].shape:
        raise ValueError("Flat and dark frames have different shapes")

    print("Hot pixels (dark frames):")
    hot = detect_hot_pixels(dark["mean"], dark["black"], hot_k, hot_min_excess)
    print("Dead/off-gain pixels (flat):")
    bad_flat = detect_flat_defects(flat["mosaic"], flat["black"], flat_k, flat_min_dev)

    mask = hot | bad_flat
    print(f"Total defective: {mask.sum()} of {mask.size} "
          f"({100 * mask.mean():.4f}%)  [hot={hot.sum()}, flat={bad_flat.sum()}]")
    np.savez_compressed(out_path, mask=mask, hot=hot, flat=bad_flat)
    return {"mask": mask, "hot": hot, "flat": bad_flat}


# ----------------------------------------------------------------------------
# Correction
# ----------------------------------------------------------------------------
def correct_defective_pixels(mosaic, mask, max_radius=3):
    """
    Replace each flagged pixel with the median of the nearest NON-flagged pixels of
    the same CFA colour (3x3 in the colour plane; grows to 5x5, 7x7 if needed).
    mosaic: HxW array (raw DN or float).  mask: HxW bool, or path to defect_map.npz.
    Returns a corrected copy with the same dtype.
    """
    if isinstance(mask, str):
        mask = np.load(mask)["mask"]
    if mask.shape != mosaic.shape:
        raise ValueError(f"mask {mask.shape} != mosaic {mosaic.shape}")
    out = mosaic.copy()
    for name, (r, c) in RGGB_OFFSETS.items():
        pm = mask[r::2, c::2]
        pv = mosaic[r::2, c::2].astype(np.float64)
        H, W = pm.shape
        new = pv.copy()
        for y, x in zip(*np.nonzero(pm)):
            for rad in range(1, max_radius + 1):
                y0, y1 = max(y - rad, 0), min(y + rad + 1, H)
                x0, x1 = max(x - rad, 0), min(x + rad + 1, W)
                good = ~pm[y0:y1, x0:x1]
                if good.any():
                    new[y, x] = np.median(pv[y0:y1, x0:x1][good])
                    break
        if np.issubdtype(mosaic.dtype, np.integer):
            new = np.clip(np.rint(new), np.iinfo(mosaic.dtype).min, np.iinfo(mosaic.dtype).max)
        out[r::2, c::2] = new.astype(mosaic.dtype)
    return out


def correct_defects_in_dng(path, mask):
    """Convenience: load a DNG, return (corrected mosaic, loaded dict)."""
    d = load_dng_mosaic(path)
    if d["pattern"] != "RGGB":
        raise ValueError(f"{path}: pattern {d['pattern']} is not RGGB")
    return correct_defective_pixels(d["mosaic"], mask), d


# ----------------------------------------------------------------------------
# Reference data
# ----------------------------------------------------------------------------
def load_reference():
    """ColorChecker24 (After November 2014): names, XYZ (Y relative, white ~0.9), illuminant xy."""
    import colour
    ccs = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
    names = list(ccs.data.keys())
    xyY = np.array([ccs.data[n] for n in names], dtype=float)
    XYZ = colour.xyY_to_XYZ(xyY)
    assert len(names) == 24, len(names)
    assert names[0].lower().startswith("dark skin"), f"patch 1 is {names[0]!r}"
    assert names[18].lower().startswith("white"), f"patch 19 is {names[18]!r}"
    return names, XYZ, ccs.illuminant


# ----------------------------------------------------------------------------
# Loading a set folder: three charts then the white field
# ----------------------------------------------------------------------------
def list_set_files(folder):
    paths = sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(".dng"))
    if len(paths) != 4:
        raise ValueError(f"Expected 3 charts + 1 white field in {folder}, found {len(paths)}: {paths}")
    print("Charts    :", [os.path.basename(p) for p in paths[:3]])
    print("White field:", os.path.basename(paths[3]))
    return paths[:3], paths[3]


def mosaic_to_rgb(mosaic_bs):
    """Black-subtracted RGGB mosaic -> half-res linear RGB (R, mean(G1,G2), B)."""
    m = mosaic_bs.astype(np.float32)
    return np.dstack([m[0::2, 0::2], 0.5 * (m[0::2, 1::2] + m[1::2, 0::2]), m[1::2, 1::2]])


def prepare_chart(path, flat_gain, defect_mask, sat_frac=0.98):
    """Defect-correct, flag saturated sites (pre-shading), lens-shade, bin to RGB."""
    d = load_dng_mosaic(path)
    if d["pattern"] != "RGGB":
        raise ValueError(f"{path}: {d['pattern']} is not RGGB")
    raw = correct_defective_pixels(d["mosaic"], defect_mask)

    # saturation map on the raw (pre-shading) data: at the white level, or on a plateau
    # at the frame maximum of that channel (sensor can saturate below the DNG white level)
    sat = np.zeros(raw.shape, dtype=bool)
    for i, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        p = raw[r::2, c::2]
        thr = min(sat_frac * d["white"], p.max())
        sat[r::2, c::2] = p >= thr
    clipmap = np.dstack([sat[0::2, 0::2], sat[0::2, 1::2] | sat[1::2, 0::2], sat[1::2, 1::2]])

    shaded = apply_lens_shading(raw, flat_gain, d["black"], d["white"])
    rgb = mosaic_to_rgb(shaded)
    return {"path": path, "rgb": rgb, "clipmap": clipmap, "white": d["white"],
            "black": d["black"], "full_scale": d["white"] - float(np.mean(d["black"]))}


# ----------------------------------------------------------------------------
# Preview + corner picking
# ----------------------------------------------------------------------------
def preview(rgb):
    s = np.percentile(rgb.reshape(-1, 3), 99.8, axis=0)
    return (np.clip(rgb / s, 0, 1)) ** (1 / 2.2)


def pick_corners(rgb):
    """Click the 4 OUTER corners of the 6x4 patch grid, clockwise from any corner.
    Needs an interactive backend (%matplotlib qt or %matplotlib widget)."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11, 8))
    plt.imshow(preview(rgb))
    plt.title("Click the 4 outer corners of the patch grid (clockwise, any start)")
    pts = plt.ginput(4, timeout=0)
    plt.close(fig)
    return [(float(x), float(y)) for x, y in pts]


# ----------------------------------------------------------------------------
# Patch sampling with orientation check
# ----------------------------------------------------------------------------
def _clockwise(corners):
    c = np.array(corners, float)
    area = 0.5 * np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1])
    return list(map(tuple, c)) if area > 0 else list(map(tuple, c[::-1]))  # image y points down


def _homography(corners_cw, k):
    c = np.array(corners_cw[k:] + corners_cw[:k], np.float32)
    dst = np.array([[0, 0], [N_COLS, 0], [N_COLS, N_ROWS], [0, N_ROWS]], np.float32)
    return cv2.getPerspectiveTransform(dst, c)          # chart units -> image px


def _patch_points(Hm, n=15, frac=0.5):
    lin = ((np.arange(n) + 0.5) / n - 0.5) * frac
    gx, gy = np.meshgrid(lin, lin)
    out = []
    for row in range(N_ROWS):
        for col in range(N_COLS):
            p = np.stack([col + 0.5 + gx.ravel(), row + 0.5 + gy.ravel()], 1).astype(np.float32)[None]
            out.append(cv2.perspectiveTransform(p, Hm)[0])
    return out


def _sample(rgb, clipmap, pts):
    h, w = rgb.shape[:2]
    means, stds, clipped = [], [], []
    for q in pts:
        xs = np.clip(np.round(q[:, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(q[:, 1]).astype(int), 0, h - 1)
        v = rgb[ys, xs]
        means.append(v.mean(0)); stds.append(v.std(0)); clipped.append(clipmap[ys, xs].mean(0))
    return np.array(means), np.array(stds), np.array(clipped)


def measure_chart(chart, corners, ref_Y, clip_tol=0.01):
    """Sample 24 patches in all 4 chart rotations, pick the one whose green channel best
    tracks the reference luminance, and report how decisive that choice was."""
    rgb, clipmap = chart["rgb"], chart["clipmap"]
    cw = _clockwise(corners)
    scores = []
    for k in range(4):
        Hm = _homography(cw, k)
        pts = _patch_points(Hm)
        m, s, c = _sample(rgb, clipmap, pts)
        scores.append((np.corrcoef(m[:, 1], ref_Y)[0, 1], k, m, s, c, pts))
    scores.sort(key=lambda t: -t[0])
    corr, k, means, stds, clipfrac, pts = scores[0]
    margin = corr - scores[1][0]

    neutral_G = means[18:24, 1]
    ramp_ok = bool(np.all(np.diff(neutral_G) < 0))
    print(f"{os.path.basename(chart['path'])}: rotation={k * 90} deg  corr(G, ref Y)={corr:.4f}  "
          f"runner-up gap={margin:.3f}  neutral ramp 19->24 decreasing={ramp_ok}")
    if corr < 0.9 or margin < 0.05 or not ramp_ok:
        print("   WARNING: numbering is not convincing. Check the overlay and your corners.")
    return {"means": means, "stds": stds, "clip_frac": clipfrac,
            "clipped": clipfrac.max(1) > clip_tol, "rotation_deg": k * 90,
            "corr": corr, "ramp_ok": ramp_ok, "pts": pts, "corners": cw, "k": k}


def plot_overlay(chart, meas, title=""):
    """Preview with the patch numbers and sampled regions. Red = clipped."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(preview(chart["rgb"]))
    for i, q in enumerate(meas["pts"]):
        col = "red" if meas["clipped"][i] else "cyan"
        ax.plot(q[:, 0], q[:, 1], ".", ms=1, color=col)
        ax.text(q[:, 0].mean(), q[:, 1].mean(), str(i + 1), color="yellow", ha="center",
                va="center", fontsize=11, weight="bold")
    c = np.array(meas["corners"] + [meas["corners"][0]])
    ax.plot(c[:, 0], c[:, 1], "-", color="lime", lw=1)
    ax.set_title(title + "  (patch 1 = dark skin, 19 = white; red = clipped)")
    return fig


# ----------------------------------------------------------------------------
# White balance gains
# ----------------------------------------------------------------------------
def fit_gains(meas, gain_patch=22):
    """Gains (R, G, B) normalised to green from one unclipped neutral patch (1-based)."""
    i = gain_patch - 1
    if meas["clipped"][i]:
        raise ValueError(f"Patch {gain_patch} is clipped; choose another neutral.")
    m = meas["means"][i]
    return np.array([m[1] / m[0], 1.0, m[1] / m[2]])


def neutral_gain_table(meas):
    """Gains from each neutral patch 19..24, to show how well they agree."""
    rows = []
    for p in range(19, 25):
        m = meas["means"][p - 1]
        rows.append((p, bool(meas["clipped"][p - 1]), m[1] / m[0], m[1] / m[2]))
    return rows


def choose_white_reference(meas_list):
    """Patch 19 if unclipped in every round compared, else the brightest unclipped neutral."""
    for p in range(19, 25):
        if not any(m["clipped"][p - 1] for m in meas_list):
            return p
    raise RuntimeError("Every neutral patch is clipped in some round.")


def apply_white_balance(rgb, gains):
    return rgb * np.asarray(gains, dtype=rgb.dtype)[None, None, :]


# ----------------------------------------------------------------------------
# Automatic corner detection (OpenCV mcc), no clicking
# ----------------------------------------------------------------------------
def _to_display_rgb8(linear_rgb):
    """Linear (1.0 = full scale) -> brightened 8-bit sRGB copy. Used only for detection."""
    x = np.clip(linear_rgb, 0, 1)
    x = np.clip(x / max(np.percentile(x, 99.5), 1e-6), 0, 1)
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    return np.ascontiguousarray((srgb * 255 + 0.5).astype(np.uint8))


def _run_mcc_detector(bgr8):
    """cv2.mcc 24-patch detector across OpenCV 4.x / 5.x signatures. Returns (detector, found)."""
    mcc24 = getattr(cv2.mcc, "MCC24", 0)
    make = getattr(cv2.mcc, "CCheckerDetector_create", None) or cv2.mcc.CCheckerDetector.create
    det = make()
    if hasattr(det, "setColorChartType"):
        try:
            det.setColorChartType(mcc24)
        except Exception:
            pass
    errors = []
    for call in (lambda: det.process(bgr8, mcc24, 1), lambda: det.process(bgr8, 1)):
        try:
            return det, bool(call())
        except cv2.error as e:
            errors.append(str(e).strip().splitlines()[0])
    raise RuntimeError("cv2.mcc process() failed with both signatures: " + " | ".join(errors))


def detect_corners(chart):
    """Outer corners (4x2) of the chart from the OpenCV detector, or None if not found.
    The corner ORDER is not trusted; measure_chart() tries all rotations and checks the
    numbering against the reference luminance."""
    x = chart["rgb"] / chart["full_scale"]
    det, found = _run_mcc_detector(np.ascontiguousarray(_to_display_rgb8(x)[..., ::-1]))
    if not found:
        return None
    box = np.asarray(det.getBestColorChecker().getBox(), dtype=np.float32).reshape(4, 2)

    # Sanity check: the detector can report a "found" chart with a degenerate box.
    h, w = chart["rgb"].shape[:2]
    area = 0.5 * abs(np.sum(box[:, 0] * np.roll(box[:, 1], -1) - np.roll(box[:, 0], -1) * box[:, 1]))
    inside = np.all(np.isfinite(box)) and box[:, 0].min() > -0.02 * w and box[:, 0].max() < 1.02 * w \
        and box[:, 1].min() > -0.02 * h and box[:, 1].max() < 1.02 * h
    if not inside or area < 0.02 * w * h or not cv2.isContourConvex(box.reshape(-1, 1, 2)):
        print(f"  detector box rejected (area {area:.0f}px, {100 * area / (w * h):.2f}% of image)")
        return None
    return [tuple(map(float, p)) for p in box]


def get_corners(chart):
    """Auto-detect; fall back to clicking if the detector finds nothing."""
    c = detect_corners(chart)
    if c is None:
        print(f"  detector found no chart in {os.path.basename(chart['path'])}; click the corners instead")
        c = pick_corners(chart["rgb"])
    return c


# ----------------------------------------------------------------------------
# Apply white balance to a raw mosaic (linear, black-subtracted) and preview it
# ----------------------------------------------------------------------------
def apply_white_balance_mosaic(mosaic_bs, gains):
    """Multiply R sites by gains[0], G sites by gains[1], B sites by gains[2] (RGGB)."""
    gR, gG, gB = [float(g) for g in gains]
    out = mosaic_bs.astype(np.float32).copy()
    out[0::2, 0::2] *= gR
    out[0::2, 1::2] *= gG
    out[1::2, 0::2] *= gG
    out[1::2, 1::2] *= gB
    return out


def to_srgb_preview(rgb, scale=None):
    """Display only. ONE common scale for all channels so the balance is preserved."""
    if scale is None:
        scale = np.percentile(rgb[..., 1], 99.5)
    x = np.clip(rgb / max(scale, 1e-6), 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)

#-----------------------------
# Demosaic
# ------------------------------
def bilinear_demosaic(mosaic, pattern):
    H, W = mosaic.shape

    # Mask of where each channel was actually measured
    masks = np.zeros((H, W, 3), dtype=bool)
    for dy in range(2):
        for dx in range(2):
            masks[dy::2, dx::2, CH[pattern[dy, dx]]] = True

    # Bilinear kernels: R/B use the 3x3 neighborhood, G uses the 4 edge neighbors
    k_rb = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float)
    k_g  = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=float)

    out = np.empty((H, W, 3))
    for c in range(3):
        k = k_g if c == 1 else k_rb
        m = masks[..., c].astype(float)
        num = convolve2d(mosaic * m, k, mode="same", boundary="fill")  # weighted sum of measured neighbors
        den = convolve2d(m, k, mode="same", boundary="fill")           # total weight of measured neighbors
        out[..., c] = num / den
    return out

def k5(d):                       # {(dy,dx): weight in 1/8 units} -> 5x5 kernel
    K = np.zeros((5, 5))
    for (dy, dx), w in d.items():
        K[2 + dy, 2 + dx] = w
    return K / 8.0

def kernel_for(pattern, dy, dx, c, with_correction=True):
    s = pattern[dy, dx]                                   # colour measured at this CFA position
    if CH[s] == c:                                        # measured: pass through
        K = np.zeros((5, 5)); K[2, 2] = 1.0
        return K
    if s != "G" and c == 1:                               # G at R / G at B   (cases 1, 2)
        b, r = BIL_G_AT_RB, COR_G_AT_RB
    elif s == "G":                                        # R or B at G       (cases 3-6)
        horiz = CH[pattern[dy, 1 - dx]] == c              # is the wanted colour left/right?
        b, r = (BIL_RB_AT_G_H, COR_RB_AT_G_H) if horiz else (BIL_RB_AT_G_V, COR_RB_AT_G_V)
    else:                                                 # R at B / B at R   (cases 7, 8)
        b, r = BIL_RB_AT_RB, COR_RB_AT_RB
    return b + r if with_correction else b
BIL_G_AT_RB   = k5({(-1,0):2, (1,0):2, (0,-1):2, (0,1):2})
BIL_RB_AT_G_H = k5({(0,-1):4, (0,1):4})              # wanted colour is left/right of the G site
BIL_RB_AT_G_V = BIL_RB_AT_G_H.T                      # wanted colour is above/below
BIL_RB_AT_RB  = k5({(-1,-1):2, (-1,1):2, (1,-1):2, (1,1):2})
COR_G_AT_RB   = k5({(0,0):4, (0,-2):-1, (0,2):-1, (-2,0):-1, (2,0):-1})
COR_RB_AT_G_H = k5({(0,0):5, (-1,-1):-1, (-1,1):-1, (1,-1):-1, (1,1):-1,
                    (0,-2):-1, (0,2):-1, (-2,0):.5, (2,0):.5})
COR_RB_AT_G_V = COR_RB_AT_G_H.T
COR_RB_AT_RB  = k5({(0,0):6, (0,-2):-1.5, (0,2):-1.5, (-2,0):-1.5, (2,0):-1.5})
def mhc_demosaic(mosaic, pattern, with_correction=True):
    H, W = mosaic.shape
    pad = np.pad(mosaic, 2, mode="reflect")               # reflect keeps the CFA phase at the border
    out = np.empty((H, W, 3))
    for dy in range(2):
        for dx in range(2):
            for c in range(3):
                K = kernel_for(pattern, dy, dx, c, with_correction)
                f = convolve2d(pad, K, mode="valid")      # one fixed linear filter, same for every pixel
                out[dy::2, dx::2, c] = f[dy::2, dx::2]
    return out


def check_measured_sites(name, out, mosaic, pattern):
    worst = 0.0
    for dy in range(2):
        for dx in range(2):
            c = CH[pattern[dy, dx]]
            d = np.abs(out[dy::2, dx::2, c] - mosaic[dy::2, dx::2]).max()
            worst = max(worst, d)
    status = "PASS" if worst <= TOL else "FAIL"
    print(f"{name:14s} largest change at measured sites: {worst:.2e}   {status}"
          f"{'  (bit-exact)' if worst == 0 else ''}")
    assert worst <= TOL, f"{name}: measured sites were altered"

# ---------------------------
# Denoising NLM
# - ----------------------------

def nlm_denoise(
    image,
    sigma,
    patch_size=7,
    search_window=21,
    h=None
):
    """
    Straightforward Non-Local Means denoising.

    Parameters
    ----------
    image : 2D numpy array
        Noisy input image.

    sigma : float
        Estimated standard deviation of additive Gaussian noise.

    patch_size : int
        Width and height of the patches used for comparison.

    search_window : int
        Width and height of the search region.

    h : float
        Filtering strength. If None, uses h = sigma.

    Returns
    -------
    denoised : 2D numpy array
        Denoised image.
    """

    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 2:
        raise ValueError("Input must be a 2D grayscale image.")

    if patch_size % 2 == 0 or search_window % 2 == 0:
        raise ValueError("patch_size and search_window must be odd.")

    if search_window < patch_size:
        raise ValueError("search_window should be at least patch_size.")

    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    if h is None:
        h = sigma

    if h <= 0:
        raise ValueError("h must be positive.")

    H, W = image.shape

    patch_radius = patch_size // 2
    search_radius = search_window // 2

    # --------------------------------------------------
    # 1. Construct Gaussian weighting kernel for patches
    # --------------------------------------------------

    coords = np.arange(-patch_radius, patch_radius + 1)

    xx, yy = np.meshgrid(coords, coords)

    # Gaussian standard deviation for spatial patch weighting
    patch_sigma = max(patch_size / 6.0, 0.5)

    gaussian_kernel = np.exp(
        -(xx**2 + yy**2) / (2 * patch_sigma**2)
    )

    # Normalize so all patch weights sum to 1
    gaussian_kernel /= gaussian_kernel.sum()

    # --------------------------------------------------
    # 2. Pad image for searching neighboring pixels
    # --------------------------------------------------

    padded = np.pad(
        image,
        pad_width=search_radius + patch_radius,
        mode="reflect"
    )

    offset = search_radius + patch_radius

    # Accumulators for weighted pixel values and weights
    weighted_sum = np.zeros_like(image, dtype=np.float32)
    weight_sum = np.zeros_like(image, dtype=np.float32)

    # --------------------------------------------------
    # 3. Compare patches at every search displacement
    # --------------------------------------------------

    for dy in range(-search_radius, search_radius + 1):

        for dx in range(-search_radius, search_radius + 1):

            # Candidate image shifted relative to target image.
            # candidate[y,x] = image[y+dy,x+dx], with reflected borders.

            candidate = padded[
                offset + dy : offset + dy + H,
                offset + dx : offset + dx + W
            ]

            # --------------------------------------------------
            # 4. Calculate squared patch distances
            # --------------------------------------------------

            squared_difference = (image - candidate)**2

            # Gaussian-weighted average squared difference
            distance = convolve(
                squared_difference,
                gaussian_kernel,
                mode="reflect"
            )

            # --------------------------------------------------
            # 5. Correct for expected noise contribution
            # --------------------------------------------------

            corrected_distance = np.maximum(
                distance - 2 * sigma**2,
                0
            )

            # --------------------------------------------------
            # 6. Convert distances into NLM weights
            # --------------------------------------------------

            weights = np.exp(
                -corrected_distance / (h**2)
            )

            # --------------------------------------------------
            # 7. Accumulate weighted center-pixel intensities
            # --------------------------------------------------

            weighted_sum += weights * candidate
            weight_sum += weights

    # --------------------------------------------------
    # 8. Normalize weighted average
    # --------------------------------------------------

    denoised = weighted_sum / weight_sum

    return denoised


#function for this 
def denoise_bayer_nlm(
    mosaic,
    pattern,
    red_sigma,
    green_sigma,
    blue_sigma
):
    """
    Denoise a Bayer mosaic by independently filtering its four CFA planes.

    Red, green, and blue use their independently optimized NLM parameters.
    The two green CFA planes are filtered separately using the same parameters.
    """

    mosaic = np.asarray(mosaic, dtype=np.float32)

    if pattern.shape != (2, 2):
        raise ValueError("Expected a 2x2 Bayer pattern.")

    denoised_mosaic = np.empty_like(mosaic)

    # Parameters for each color
    params = {
        "R": {
            "sigma": red_sigma,
            "h": best_red_h,
            "patch_size": best_red_patch_size,
            "search_window": best_red_window_size
        },
        "G": {
            "sigma": green_sigma,
            "h": best_h,
            "patch_size": best_patch_size,
            "search_window": best_window_size
        },
        "B": {
            "sigma": blue_sigma,
            "h": best_blue_h,
            "patch_size": best_blue_patch_size,
            "search_window": best_blue_window_size
        }
    }

    # Extract and denoise each of the four CFA subplanes
    for dy in range(2):
        for dx in range(2):

            color = pattern[dy, dx]
            p = params[color]

            plane = mosaic[dy::2, dx::2]

            denoised_plane = nlm_denoise(
                plane,
                sigma=p["sigma"],
                patch_size=p["patch_size"],
                search_window=p["search_window"],
                h=p["h"]
            )

            denoised_mosaic[dy::2, dx::2] = denoised_plane

    return denoised_mosaic



# ------------------------------------------------------------
# 2. Helper functions for metadata extraction
# ------------------------------------------------------------

def read_dng_metadata(fname):
    """
    Read TIFF/DNG tags and EXIF metadata using tifffile.
    The EXIF IFD is returned as a dictionary.
    """
    tags = {}

    with tifffile.TiffFile(fname) as tif:
        page = tif.pages[0]

        # Main TIFF/DNG tags
        for tag in page.tags.values():
            tags[tag.name] = tag.value

        # EXIF IFD (tag 34665)
        exif_tag = page.tags.get(34665)

        if exif_tag is not None:
            exif_dict = exif_tag.value

            if isinstance(exif_dict, dict):
                tags.update(exif_dict)

    return tags


def rational_to_float(value):
    """Convert TIFF/EXIF rational tuples to floating-point values."""
    if isinstance(value, (tuple, list)) and len(value) == 2:
        numerator, denominator = value

        if denominator != 0:
            return numerator / denominator

    return value


def get_tag(tags, *names):
    """Return the first available tag from a list of possible names."""
    for name in names:
        if name in tags:
            return tags[name]

    return "Not recorded"


def format_exposure(value):
    """Format exposure time as a fraction when available."""
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return f"{value[0]}/{value[1]} s"

    if isinstance(value, (int, float)):
        return f"{value:.6f} s"

    return str(value)

def get_declared_white_level(fname):
    with tifffile.TiffFile(fname) as tif:
        page = tif.pages[0]
        tag = page.tags.get("WhiteLevel")

        if tag is None:
            return np.nan

        value = tag.value

        if isinstance(value, (tuple, list, np.ndarray)):
            return float(np.max(value))

        return float(value)



def dng_list(folder):
    return sorted(glob.glob(os.path.join(folder, "*.[dD][nN][gG]")))

def load_linear_quarter(path, sat):
    """Black-subtracted, saturation-scaled, non-demosaiced quarter-res RGB (CFA read from file)."""
    with rawpy.imread(path) as raw:
        mosaic = raw.raw_image_visible.astype(np.float64)
        colors = raw.raw_colors_visible.astype(np.int64)
        desc = raw.color_desc.decode("ascii")
        black = np.array(raw.black_level_per_channel, dtype=np.float64)
    H, W = mosaic.shape
    H2, W2 = H - H % 2, W - W % 2
    mosaic, colors = mosaic[:H2, :W2], colors[:H2, :W2]
    lin = (mosaic - black[colors]) / (sat - black[colors])
    letters = np.array(list(desc))[colors]
    blk = lambda a: a.reshape(H2//2, 2, W2//2, 2).transpose(0, 2, 1, 3).reshape(H2//2, W2//2, 4)
    lb, cb = blk(lin), blk(letters)
    out = np.zeros((H2//2, W2//2, 3))
    for ch, L in enumerate("RGB"):
        m = (cb == L)
        out[..., ch] = (lb * m).sum(axis=2) / m.sum(axis=2)
    return out

def normalize_to_center(flat, frac=CENTER_FRAC):
    """Divide each channel by its own mean over a central window."""
    h, w, _ = flat.shape
    half = max(8, int(frac * min(h, w)))
    cy, cx = h // 2, w // 2
    center = flat[cy-half:cy+half, cx-half:cx+half].reshape(-1, 3).mean(axis=0)
    return flat / center, center

def h_profile(norm, band=PROFILE_BAND):
    h = norm.shape[0]
    cy = h // 2
    return norm[cy-band:cy+band].mean(axis=0)   # (width, 3)


def _to_display_rgb8(linear_rgb):
    """Linear -> brightened 8-bit sRGB copy. Used ONLY for detection and for drawing."""
    x = np.clip(linear_rgb, 0, 1)
    x = np.clip(x / max(np.percentile(x, 99.5), 1e-6), 0, 1)
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    return np.ascontiguousarray((srgb * 255 + 0.5).astype(np.uint8))


def sample_patch_means(img, corners, core=CORE):
    """Mean / std of the central part of each of the 24 patches, in the ORIGINAL (linear) img.

    corners: 4x2 outer corners of the patch grid, order top-left, top-right, bottom-right,
    bottom-left as the chart reads. A homography maps the 6x4 grid onto them, so tilted
    or rotated charts work. `core` is the fraction of each patch side that is sampled.
    """
    assert 0 < core <= 0.5, "sample at most the middle half of each patch"
    unit = np.float32([[0, 0], [GRID_COLS, 0], [GRID_COLS, GRID_ROWS], [0, GRID_ROWS]])
    M = cv2.getPerspectiveTransform(unit, np.float32(corners))
    h, w = img.shape[:2]
    means, stds, clipped, counts, polys = [], [], [], [], []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cx, cy, d = c + 0.5, r + 0.5, core / 2
            q = np.float32([[cx - d, cy - d], [cx + d, cy - d], [cx + d, cy + d], [cx - d, cy + d]])
            poly = cv2.perspectiveTransform(q[None], M)[0]
            pts = np.round(poly).astype(np.int32)
            x, y, bw, bh = cv2.boundingRect(pts)
            if x < 0 or y < 0 or x + bw > w or y + bh > h:
                raise ValueError(f"patch {len(means) + 1} falls outside the image")
            mask = np.zeros((bh, bw), np.uint8)
            cv2.fillConvexPoly(mask, pts - np.array([x, y], dtype=np.int32), 1)
            px = img[y:y + bh, x:x + bw][mask.astype(bool)].astype(np.float64)
            means.append(px.mean(axis=0))
            stds.append(px.std(axis=0))
            clipped.append(float((px >= 0.999).any(axis=1).mean()))
            counts.append(len(px))
            polys.append(poly)
    return (np.array(means), np.array(stds), np.array(clipped), np.array(counts), polys)


def _run_mcc_detector(bgr8):
    """Run cv2.mcc for the 24-patch chart on an 8-bit BGR image, across OpenCV versions.

    OpenCV 4.x:  process(image, chartType, nc)
    OpenCV 5.x:  chart type is a setting (setColorChartType) and process(image[, nc])
    Returns (detector, found).
    """
    mcc24 = getattr(cv2.mcc, "MCC24", 0)
    make = getattr(cv2.mcc, "CCheckerDetector_create", None) or cv2.mcc.CCheckerDetector.create
    det = make()
    if hasattr(det, "setColorChartType"):
        try:
            det.setColorChartType(mcc24)
        except Exception:
            pass
    errors = []
    for call in (lambda: det.process(bgr8, mcc24, 1),       # OpenCV 4.x signature
                 lambda: det.process(bgr8, 1)):             # OpenCV 5.x signature
        try:
            return det, bool(call())
        except cv2.error as e:
            errors.append(str(e).strip().splitlines()[0])
    raise RuntimeError("cv2.mcc process() failed with both signatures: " + " | ".join(errors))


def find_colorchecker(linear_rgb, corners=None, core=CORE):
    """Locate the 24 patches and measure their mean linear RGB.

    corners: optional manual outer corners (TL, TR, BR, BL); if None the cv2.mcc detector
    is used on an 8-bit display copy. The linear image is what actually gets measured.
    Returns a dict: means (24,3), stds, clipped_frac, n_pixels, polys, corners,
    orientation_rms (detector agreement, 8-bit counts; nan for manual corners),
    grey_ramp_ok (patches 19..24 get darker).
    """
    rgb8 = _to_display_rgb8(linear_rgb)
    orientation_rms = float("nan")

    if corners is None:
        det, found = _run_mcc_detector(np.ascontiguousarray(rgb8[..., ::-1]))
        if not found:
            raise RuntimeError("no 24-patch chart found; pass corners=[TL, TR, BR, BL] manually")
        checker = det.getBestColorChecker()
        box = np.asarray(checker.getBox(), dtype=np.float32).reshape(4, 2)

        # The detector's own patch means (8-bit RGB), used to verify the corner order.
        try:
            ref = np.asarray(checker.getChartsRGB())[:, 1].reshape(-1, 3)
            assert len(ref) == 24
        except Exception:
            ref = None

        if ref is None:
            print("[warn] could not read the detector's patch colours; orientation is NOT "
                  "verified. Check the figure and the grey-ramp line carefully.")
            corners = box
        else:
            # Corner order from the detector is not guaranteed: try 4 rotations and the
            # reversed winding, keep the one whose patch colours match the detector's own.
            best = None
            for pts in (box, box[::-1]):
                for s in range(4):
                    cand = np.roll(pts, -s, axis=0)
                    m, *_ = sample_patch_means(rgb8, cand, core)
                    err = float(np.sqrt(np.mean((m - ref) ** 2)))
                    if best is None or err < best[1]:
                        best = (cand, err)
            corners, orientation_rms = best

    corners = np.float32(corners)
    means, stds, clipped, counts, polys = sample_patch_means(linear_rgb, corners, core)
    lum = means @ np.array([0.2126, 0.7152, 0.0722])
    return {"means": means, "stds": stds, "clipped_frac": clipped, "n_pixels": counts,
            "polys": polys, "corners": corners, "orientation_rms": orientation_rms,
            "grey_ramp_ok": bool(np.all(np.diff(lum[18:24]) < 0)), "core": core}


def plot_sampled_regions(linear_rgb, result, title="", save=None):
    """Draw the sampled regions (yellow, numbered) and chart outline (cyan) on the chart."""
    rgb8 = _to_display_rgb8(linear_rgb)
    corners = result["corners"]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(rgb8)
    ax.add_patch(plt.Polygon(corners, closed=True, fill=False, ec="cyan", lw=1.5, ls="--"))
    for i, poly in enumerate(result["polys"]):
        ax.add_patch(plt.Polygon(poly, closed=True, fill=False, ec="yellow", lw=1.3))
        ax.text(*poly.mean(axis=0), str(i + 1), color="yellow", ha="center", va="center",
                fontsize=9, fontweight="bold")
    (x0, y0), (x1, y1) = corners.min(axis=0), corners.max(axis=0)
    m = 0.05 * max(x1 - x0, y1 - y0)
    h, w = rgb8.shape[:2]
    ax.set_xlim(max(x0 - m, 0), min(x1 + m, w))
    ax.set_ylim(min(y1 + m, h), max(y0 - m, 0))
    ax.set_title(f"{title}\nyellow = sampled regions (central {result['core']:.0%} of each patch), "
                 f"cyan = detected chart outline", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=200)
    return fig


def patch_table(result):
    """24 x 3 DataFrame of mean linear R, G, B (black-subtracted, 1.0 = saturation)."""
    t = pd.DataFrame(result["means"], columns=["R", "G", "B"],
                     index=[f"{i + 1} {n}" for i, n in enumerate(PATCH_NAMES)])
    t.index.name = "patch"
    return t


def apply_color_correction(
    rgb_camera,
    camera_to_xyz,
    source_illuminant,
    exposure_scale=1.0
):
    """
    Apply a fitted camera RGB -> XYZ matrix and convert XYZ to sRGB.

    Parameters
    ----------
    rgb_camera : ndarray
        Linear, white-balanced camera RGB image (H, W, 3).

    camera_to_xyz : ndarray
        3x3 camera RGB-to-XYZ matrix.
        Uses the convention: XYZ = RGB @ camera_to_xyz.T

    source_illuminant : ndarray
        xy chromaticity of the illuminant used to generate
        the reference ColorChecker XYZ values.

    exposure_scale : float
        Optional scalar exposure adjustment. Default = 1.0.

    Returns
    -------
    rgb_srgb : ndarray
        Color-corrected, gamma-encoded sRGB image.
    xyz : ndarray
        Color-corrected XYZ image before sRGB conversion.
    """

    rgb_camera = np.asarray(rgb_camera, dtype=np.float64)
    camera_to_xyz = np.asarray(camera_to_xyz, dtype=np.float64)

    if rgb_camera.ndim != 3 or rgb_camera.shape[-1] != 3:
        raise ValueError("Input image must have shape (H, W, 3).")

    if camera_to_xyz.shape != (3, 3):
        raise ValueError("camera_to_xyz must have shape (3, 3).")

    # Apply the fitted camera RGB -> XYZ matrix
    xyz = rgb_camera @ camera_to_xyz.T

    # Optional exposure adjustment
    xyz *= exposure_scale

    # Convert XYZ to sRGB.
    # colour.XYZ_to_sRGB performs chromatic adaptation from
    # the supplied source illuminant to the sRGB D65 white point.
    rgb_srgb = colour.XYZ_to_sRGB(
        xyz,
        illuminant=source_illuminant
    )

    return rgb_srgb, xyz



def apply_gamma_curve(rgb, gamma=1.0):
    """
    Apply a gamma tone curve to an sRGB image.

    Input and output are normalized to [0, 1].
    """

    if gamma <= 0:
        raise ValueError("Gamma must be positive.")

    rgb = np.asarray(rgb, dtype=np.float64)

    # Clip out-of-gamut values before tone adjustment
    rgb = np.clip(rgb, 0, 1)

    return rgb ** gamma


def apply_s_curve(rgb, strength=6.0):
    """
    Apply a normalized logistic S-curve to an sRGB image.

    strength controls contrast:
        0 or near 0: approximately linear
        larger values: stronger contrast
    """

    rgb = np.asarray(rgb, dtype=np.float64)

    if strength <= 0:
        raise ValueError("Strength must be positive.")

    rgb = np.clip(rgb, 0, 1)

    # Logistic curve
    curve = 1 / (1 + np.exp(-strength * (rgb - 0.5)))

    # Normalize endpoints to exactly 0 and 1
    low = 1 / (1 + np.exp(strength / 2))
    high = 1 / (1 + np.exp(-strength / 2))

    curve = (curve - low) / (high - low)

    return curve

def unsharp_mask(rgb, sigma=1.0, amount=0.5):
    """
    Fast unsharp masking using OpenCV GaussianBlur.
    """
    # float32 is sufficient for display-oriented sharpening
    rgb = np.asarray(rgb, dtype=np.float32)

    # Gaussian blur
    blurred = cv2.GaussianBlur(
        rgb,
        ksize=(0, 0),       # Automatically determine kernel size from sigma
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT
    )

    # Unsharp mask
    sharpened = rgb + amount * (rgb - blurred)

    return np.clip(sharpened, 0, 1)
