"""Map-to-Product pipeline for The Natural Archive (v2.6, GitHub Actions edition).

Ported from the CodeWords service map_pipeline_06db0b4d. Same algorithms and outputs.

mode=assess: fetch a public-domain map scan, verify it, and measure fold, border lines,
margins, edge tone and size options. mode=build: non-generative restoration (fold
lighting flatten, neatline dewarp, smear trim), paper extension to exact frame ratios,
numeric QA, one QA contact sheet, and direct upload of every file to signed URLs.

Usage: python pipeline.py request.json result.json
"""
import hashlib
import io
import json
import sys
import time
import traceback
import urllib.parse
from typing import Any, Dict, List, Literal, Optional

import cv2
import httpx
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from scipy.ndimage import binary_dilation, gaussian_filter, gaussian_filter1d, median_filter

Image.MAX_IMAGE_PIXELS = None
UA = "TheNaturalArchiveMapBot/1.0 (https://thenaturalarchive-shop.fourthwall.com) python-httpx"


class PipelineError(Exception):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status
        self.detail = detail


HTTPException = PipelineError


class _Log:
    def info(self, msg, **kw):
        extra = " ".join(f"{k}={v}" for k, v in kw.items())
        print(f"[{time.strftime('%H:%M:%S')}] {msg} {extra}", flush=True)


logger = _Log()


def commons_url(name: str) -> str:
    n = name.replace(" ", "_")
    m = hashlib.md5(n.encode()).hexdigest()
    return f"https://upload.wikimedia.org/wikipedia/commons/{m[0]}/{m[:2]}/{urllib.parse.quote(n)}"


def fetch(name: Optional[str], url: Optional[str]):
    src = url or commons_url(name)
    last = None
    for attempt in range(4):
        try:
            r = httpx.get(src, headers={"User-Agent": UA}, follow_redirects=True, timeout=300)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            b = r.content
            a = np.asarray(Image.open(io.BytesIO(b)).convert("RGB")).copy()
            return b, a
        except httpx.HTTPError as e:
            last = e
            time.sleep(5 * (attempt + 1))
    raise PipelineError(502, f"fetch failed for {src}: {last or 'HTTP 429'}")


def gray(a):
    return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)


def closed(g, k=45):
    return cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def col_paper(g):
    return np.percentile(closed(g), 85, axis=0).astype(float)


def find_fold(g):
    colp = gaussian_filter1d(col_paper(g), 2)
    W = g.shape[1]
    c = W // 2
    lo, hi = c - W // 10, c + W // 10
    d = np.diff(colp[lo:hi])
    xf = int(np.argmax(np.abs(d)) + lo)
    step = float(abs(colp[min(W - 1, xf + 15)] - colp[max(0, xf - 15)]))
    left = float(np.median(colp[max(0, xf - 600):xf - 300]) - colp[max(0, xf - 20)]) if xf > 620 else 0.0
    right = float(np.median(colp[xf + 300:xf + 600]) - colp[min(W - 1, xf + 20)]) if xf + 600 < W else 0.0
    return xf, step, max(left, right, 0.0)


def border_lines(g):
    H, W = g.shape
    d = (g < 110).astype(np.float32)
    rf = gaussian_filter1d(d[:, W // 10:9 * W // 10].mean(1), 1)
    cf = gaussian_filter1d(d[H // 10:9 * H // 10].mean(0), 1)

    def first_peak(v, rng):
        seg = v[rng]
        thr = max(0.15, 0.5 * seg.max())
        idx = np.where(seg >= thr)[0]
        return None if len(idx) == 0 else int(np.arange(len(v))[rng][idx[0]])

    top = first_peak(rf, slice(0, H // 12))
    bot = first_peak(rf[::-1], slice(0, H // 12))
    left = first_peak(cf, slice(0, W // 12))
    right = first_peak(cf[::-1], slice(0, W // 12))
    return (top, None if bot is None else H - 1 - bot, left, None if right is None else W - 1 - right)


def track(g, x_start, y_start, win=4):
    H, W = g.shape
    ys = np.zeros(W, np.float32)

    def go(rng):
        y = y_start
        for x in rng:
            lo, hi = int(max(0, y - win)), int(min(H, y + win + 1))
            y = lo + int(np.argmin(g[lo:hi, max(0, x - 3):x + 4].mean(1)))
            ys[x] = y

    go(range(x_start, W))
    go(range(x_start, -1, -1))
    return ys


def trace_line(g, y_est, xf):
    W = g.shape[1]
    ys = np.zeros(W, np.float32)
    lo, hi = max(0, y_est - 16), min(g.shape[0], y_est + 17)
    for xs, sl in ((int(0.2 * W), slice(0, xf)), (min(W - 10, xf + int(0.15 * W)), slice(xf, W))):
        y0 = lo + int(np.argmin(g[lo:hi, xs - 3:xs + 4].mean(1)))
        ys[sl] = track(g, xs, y0)[sl]
    return ys


def side_smooth(v, xf, W, edge=100):
    d = v.copy()
    d[:edge] = d[edge]
    d[W - edge:] = d[W - edge - 1]
    out = []
    for seg, end in ((d[:xf], True), (d[xf:], False)):
        seg = median_filter(seg, 31, mode="nearest")
        s4, s40 = gaussian_filter1d(seg, 4, mode="nearest"), gaussian_filter1d(seg, 40, mode="nearest")
        dist = (len(seg) - 1 - np.arange(len(seg))) if end else np.arange(len(seg))
        w = np.clip(1 - (dist - 60) / 100, 0, 1)
        out.append(w * s4 + (1 - w) * s40)
    return np.concatenate(out)


def paper_field(a):
    H, W = a.shape[:2]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    out = np.zeros((H, W, 3), np.float32)
    for ch in range(3):
        cl = cv2.morphologyEx(a[..., ch], cv2.MORPH_CLOSE, k)
        coarse = np.stack([np.percentile(cl[max(0, y - 100):min(H, y + 101)], 85, axis=0)
                           for y in range(0, H, 30)]).astype(np.float32)
        out[..., ch] = cv2.resize(coarse, (W, H), interpolation=cv2.INTER_LINEAR)
    return out


def flatten(a, xf):
    H, W = a.shape[:2]
    F = paper_field(a)
    narrow, wide = gaussian_filter(F, (25, 1.5, 0)), gaussian_filter(F, (40, 25, 0))
    w = np.clip(1 - (np.abs(np.arange(W) - xf).astype(np.float32) - 90) / 60, 0, 1)[None, :, None]
    S = w * narrow + (1 - w) * wide
    lx = S[H // 10:9 * H // 10, max(0, xf - int(0.3 * W)):xf - 300].reshape(-1, 3)
    rx = S[H // 10:9 * H // 10, xf + 300:min(W, xf + int(0.3 * W))].reshape(-1, 3)
    ref = np.median(rx if rx.mean() >= lx.mean() else lx, axis=0)
    return np.clip(a.astype(np.float32) * (ref[None, None, :] / np.maximum(S, 1)), 0, 255)


def dewarp(img, y_top, y_bot, xf):
    H, W = img.shape[:2]
    g = gray(np.clip(img, 0, 255).astype(np.uint8))
    top, bot = trace_line(g, y_top, xf), trace_line(g, y_bot, xf)
    ref = slice(min(W - 2, xf + 300), W - 100) if W - 100 - (xf + 300) > 400 else slice(100, max(101, xf - 300))
    tt, tb = float(np.median(top[ref])), float(np.median(bot[ref]))
    dt, db = side_smooth(top - tt, xf, W), side_smooth(bot - tb, xf, W)
    yy = np.arange(H, dtype=np.float32)[:, None]
    t = np.clip((yy - tt) / (tb - tt), 0, 1)
    dy = dt[None, :] * (1 - t) + db[None, :] * t
    mapx = np.tile(np.arange(W, dtype=np.float32), (H, 1))
    out = cv2.remap(img.astype(np.float32), mapx, (yy + dy).astype(np.float32), cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REPLICATE)
    trim_t = int(np.ceil(max(0, -dt.min()))) + 2
    trim_b = int(np.ceil(max(0, db.max()))) + 2
    return np.clip(out, 0, 255), trim_t, trim_b, float(np.ptp(top[150:W - 150])), float(np.ptp(bot[150:W - 150]))


def auto_trim_edges(a, cap_frac=0.12, inset=4):
    """Trim non-paper background (copy stand, colour targets, book edges) off each scan edge.
    Walks inward until a line is clean paper; never walks into the map (capped near printed borders)."""
    g = gray(a)
    H, W = g.shape
    paper = float(np.median(np.percentile(g[H // 4:3 * H // 4:4, W // 4:3 * W // 4], 85, axis=0)))
    thr = paper - 50
    capx, capy = int(cap_frac * W), int(cap_frac * H)

    def walk(stat, cap):
        idx = np.where(stat[:cap] > thr)[0]
        return None if len(idx) == 0 else int(idx[0])

    colp2 = np.percentile(g[H // 10:9 * H // 10:4], 2, axis=0)
    l, r = walk(colp2, capx), walk(colp2[::-1], capx)
    l0, r0 = l or 0, r or 0
    rowp2 = np.percentile(g[:, l0:W - r0:4], 2, axis=1)
    t, b = walk(rowp2, capy), walk(rowp2[::-1], capy)
    t0, b0 = t or 0, b or 0
    colp2b = np.percentile(g[t0:H - b0:4], 2, axis=0)
    l, r = walk(colp2b, capx), walk(colp2b[::-1], capx)
    flags = [s for s, v in zip("tblr", (t, b, l, r)) if v is None]
    trims = [0 if v is None else (v + inset if v > 0 else 0) for v in (t, b, l, r)]
    bt, bb, bl, br = border_lines(g)
    dist = [bt, None if bb is None else H - 1 - bb, bl, None if br is None else W - 1 - br]
    for i, (s, d) in enumerate(zip("tblr", dist)):
        if d is not None and d > 10 and trims[i] > d - 8:
            trims[i] = d - 8
            flags.append(f"{s}:capped_at_border")
    return trims, flags, round(paper, 1)


def neatline_angle(g, n=17, band_frac=0.12, strip_frac=0.02):
    """Skew of the printed border (neatline) relative to the scan, in degrees (positive = clockwise in image
    coordinates). For each side, narrow strips walk in from the scan edge to the first dark line; a robust line
    fit through those points gives that side's angle. Returns (angle, per-side angles, spread) or None if no
    border is found on at least two sides."""
    H, W = g.shape
    dark = (g < 110).astype(np.float32)
    out = {}
    diag = {}
    neatline_angle.diag = diag

    def side_pts(img, axis_len, cross_len):
        pts = []
        sw = max(5, int(strip_frac * cross_len))
        lim = int(band_frac * axis_len)
        for c in np.linspace(0.15, 0.85, n):
            x = int(c * cross_len)
            prof = img[:lim, max(0, x - sw // 2):x + sw // 2 + 1].mean(1)
            prof = gaussian_filter1d(prof, 1)
            light = np.where(prof < 0.2)[0]
            if not len(light):
                continue
            s0 = int(light[0])  # skip dark scanner background at the very edge
            seg = prof[s0:]
            if not len(seg) or seg.max() < 0.3:
                continue
            idx = np.where(seg >= max(0.3, 0.6 * seg.max()))[0]
            pts.append((x, float(s0 + idx[0])))
        return pts

    views = {"t": dark, "b": dark[::-1], "l": dark.T, "r": dark.T[::-1]}
    for k, img in views.items():
        axis_len, cross_len = img.shape
        pts = side_pts(img, axis_len, cross_len)
        diag[k] = {"pts": len(pts)}
        if len(pts) < 0.6 * n:
            continue
        x = np.array([p[0] for p in pts], float)
        y = np.array([p[1] for p in pts], float)
        keep = np.ones(len(x), bool)
        for _ in range(3):
            a, b = np.polyfit(x[keep], y[keep], 1)
            r = y - (a * x + b)
            mad = float(np.median(np.abs(r[keep]))) + 0.5
            keep = np.abs(r) <= 3 * mad + 2
            if keep.sum() < 0.6 * n:
                break
        diag[k].update(kept=int(keep.sum()), mad=round(float(np.median(np.abs(r[keep]))), 2))
        if keep.sum() < 0.6 * n or float(np.median(np.abs(r[keep]))) > 4:
            continue
        a, b = np.polyfit(x[keep], y[keep], 1)
        ang = float(np.degrees(np.arctan(a)))
        # convert each side's slope to one rotation sense (clockwise positive in image coords)
        out[k] = {"t": ang, "b": -ang, "l": -ang, "r": ang}[k]
    if len(out) < 2:
        return None
    vals = np.array(list(out.values()))
    return float(np.median(vals)), {k: round(v, 3) for k, v in out.items()}, float(np.ptp(vals))


def deskew(a, angle):
    """Rotate the whole scan so the printed border is level, then crop the wedges the rotation leaves at the
    corners so no invented pixels remain. Resampling only (Lanczos); nothing is drawn."""
    H, W = a.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, 1.0)
    rot = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    t = abs(np.tan(np.radians(angle)))
    cy, cx = int(np.ceil(W * t / 2)) + 2, int(np.ceil(H * t / 2)) + 2
    return np.ascontiguousarray(rot[cy:H - cy, cx:W - cx]), [cy, cy, cx, cx]


def margin_boxes(shape, borders, inset=4, gap=8, min_len=10):
    H, W = shape[:2]
    t, b, l, r = borders
    boxes = []
    if t and t - gap - inset >= min_len:
        boxes.append((inset, t - gap, W // 10, 9 * W // 10, False))
    if b and H - inset - (b + gap) >= min_len:
        boxes.append((b + gap, H - inset, W // 10, 9 * W // 10, False))
    if l and l - gap - inset >= min_len:
        boxes.append((H // 10, 9 * H // 10, inset, l - gap, True))
    if r and W - inset - (r + gap) >= min_len:
        boxes.append((H // 10, 9 * H // 10, r + gap, W - inset, True))
    return boxes


def band_split(p):
    f = p - gaussian_filter(p, (2.5, 2.5, 0))
    m = gaussian_filter(p, (2.5, 2.5, 0)) - gaussian_filter(p, (12, 12, 0))
    return f, m


def grain_sources(a, boxes, adaptive=False):
    """Blank-margin grain tiles. adaptive=True raises thresholds to 4x each box's median detail (camera photos)."""
    srcs, masks, tf, tm = [], [], [], []
    for (y0, y1, x0, x1, tr) in boxes:
        p = a[y0:y1, x0:x1].astype(np.float32)
        if tr:
            p = np.ascontiguousarray(p.transpose(1, 0, 2))
        f, m = band_split(p)
        af, am = np.abs(f).max(2), np.abs(m).max(2)
        kf, km = 7.0, 10.0
        if adaptive:
            kf, km = max(7.0, 4.0 * float(np.median(af))), max(10.0, 4.0 * float(np.median(am)))
        bad = binary_dilation(af > kf, iterations=2) | binary_dilation(am > km, iterations=3)
        good = ~bad
        if good.mean() < 0.3:
            continue
        tf.append(float(f[good].std()))
        tm.append(float(m[good].std()))
        hp = f + m
        hp = np.where(good[..., None], hp, 0)
        hp -= hp.mean(1, keepdims=True)
        srcs.append(hp.astype(np.float32))
        masks.append(bad)
    if not srcs:
        raise HTTPException(422, "no usable blank margin for paper grain")
    return srcs, masks, float(np.median(tf)), float(np.median(tm))


def blank_boxes(a, seed=1, max_n=48, bh=64, bw=256, step=16):
    """Clean blank-paper windows anywhere on a sheet (margins, blank sea or land, a donor page)."""
    H, W = a.shape[:2]
    if H < bh + 8 or W < bw + 8:
        return []
    g8 = gray(a)
    g = g8.astype(np.float32)
    cl = closed(g8, 31).astype(np.float32)
    f = g - gaussian_filter(g, 2.5)
    m = gaussian_filter(g, 2.5) - gaussian_filter(g, 12)
    det = gaussian_filter(np.abs(f) + np.abs(m), 3)
    thr = max(6.0, 2.5 * float(np.median(det)))
    del f, m
    rgb = a.astype(np.int16)
    chroma = (rgb.max(2) - rgb.min(2)).astype(np.float32)
    del rgb
    bright = g > np.percentile(g, 60)
    pc = float(np.median(chroma[bright]))
    pl = float(np.percentile(g, 90))
    bad = (det > thr) | (g < cl - 20) | (np.abs(chroma - pc) > 24) | (g < pl - 90)
    del det, chroma, cl
    bad = binary_dilation(bad, iterations=2)
    ii = np.zeros((H + 1, W + 1), np.int32)
    ii[1:, 1:] = bad.astype(np.int32).cumsum(0).cumsum(1)
    del bad
    xs = np.arange(4, W - bw - 4, step)
    cand = []
    lim = 0.03 * bh * bw
    for y in range(4, H - bh - 4, step):
        y1 = y + bh
        s = ii[y1, xs + bw] - ii[y, xs + bw] - ii[y1, xs] + ii[y, xs]
        cand += [(y, int(x)) for x in xs[s < lim]]
    rng = np.random.default_rng(seed)
    rng.shuffle(cand)
    taken = []
    for (y, x) in cand:
        if all(abs(y - ty) >= bh or abs(x - tx) >= bw for ty, tx in taken):
            taken.append((y, x))
            if len(taken) >= max_n:
                break
    return [(y, y + bh, x, x + bw, False) for y, x in taken]


def rich_grain(R, bR, req, res):
    """Grain pool: thick margins, clean interior blank patches, optional donor page. Grain only, never tone."""
    pools = []
    thick = [bx for bx in margin_boxes(R.shape, bR) if ((bx[3] - bx[2]) if bx[4] else (bx[1] - bx[0])) >= 60]
    if thick:
        pools.append(("margin", R, thick))
    if req.grain_mode == "rich":
        pools.append(("interior", R, blank_boxes(R, req.seed)))
    if req.grain_file or req.grain_url:
        _, D = fetch(req.grain_file, req.grain_url)
        if req.grain_scale != 1.0:
            D = cv2.resize(D, (int(round(D.shape[1] * req.grain_scale)), int(round(D.shape[0] * req.grain_scale))),
                           interpolation=cv2.INTER_AREA)
        pools.append(("donor", D, blank_boxes(D, req.seed)))
    srcs, masks, tfs, tms, used = [], [], [], [], {}
    for name, img, boxes in pools:
        if not boxes:
            used[name] = 0
            continue
        try:
            s, mk, tf, tm = grain_sources(img, boxes, adaptive=True)
        except HTTPException:
            used[name] = 0
            continue
        srcs += s
        masks += mk
        tfs.append(tf)
        tms.append(tm)
        used[name] = len(s)
    res["grain_pool"] = used
    if len(srcs) < 3:
        raise HTTPException(422, f"rich grain: too few clean blank patches {used}")
    return srcs, masks, float(np.median(tfs)), float(np.median(tms))


def quilt(srcs, masks, n, W, seed, th=24, tw=256, oy=8, ox=32):
    rng = np.random.default_rng(seed)
    th = min(th, min(s.shape[0] for s in srcs))
    oy = max(oy, th // 4) if th > 24 else oy
    oy = min(oy, th // 3)
    acc = np.zeros((n + th, W + tw, 3), np.float32)
    ws = np.zeros((n + th, W + tw, 1), np.float32)
    ry = np.minimum(np.arange(th) + 1, np.arange(th)[::-1] + 1).clip(max=oy) / oy
    rx = np.minimum(np.arange(tw) + 1, np.arange(tw)[::-1] + 1).clip(max=ox) / ox
    wm = (ry[:, None] * rx[None, :])[..., None].astype(np.float32)
    for y in range(0, n, th - oy):
        for x in range(-int(rng.integers(tw)), W, tw - ox):
            for _ in range(25):
                i = int(rng.integers(len(srcs)))
                s, mk = srcs[i], masks[i]
                r0 = int(rng.integers(0, s.shape[0] - th + 1))
                c0 = int(rng.integers(0, max(1, s.shape[1] - tw + 1)))
                if mk[r0:r0 + th, c0:c0 + tw].mean() < 0.02:
                    break
            t = s[r0:r0 + th, c0:c0 + tw]
            if t.shape[1] < tw:
                t = np.concatenate([t] * (tw // t.shape[1] + 1), 1)[:, :tw]
            t = t - t.mean(1, keepdims=True)  # removes horizontal lines (plate marks, chain lines) inside the tile
            if rng.random() < .5:
                t = t[:, ::-1]
            if rng.random() < .5:
                t = t[::-1]
            xa = max(x, 0)
            acc[y:y + th, xa:x + tw] += (t * wm)[:, xa - x:]
            ws[y:y + th, xa:x + tw] += wm[:, xa - x:]
    fld = (acc / np.maximum(ws, 1e-6))[:n, :W]
    return fld - fld.mean(1, keepdims=True)


def calibrated(fld, tf, tm):
    f, m = band_split(fld)
    sf, sm = max(float(f.std()), 1e-6), max(float(m.std()), 1e-6)
    return f * (tf / sf) + m * (tm / sm)


def ext_rows(a, n, where, srcs, masks, tf, tm, seed, ease, tone_mode="margin", th=24):
    """Rows of added paper beyond one edge. tone_mode 'edge' (v2.4), 'margin' (default), 'smooth' (v2.6)."""
    if n <= 0:
        return np.zeros((0, a.shape[1], 3), np.float32)
    W = a.shape[1]
    if tone_mode == "edge":
        edge = np.median(a[:2] if where == "top" else a[-2:], axis=0).astype(np.float32)
        e3 = gaussian_filter1d(edge, 3, axis=0, mode="nearest")
        efar = gaussian_filter1d(edge, 40, axis=0, mode="nearest")
        ramp = ease
    elif tone_mode == "smooth":
        raw = np.median(a[:2] if where == "top" else a[-2:], axis=0).astype(np.float32)
        med = median_filter(raw, size=(151, 1), mode="nearest")
        edge = np.where(np.abs(raw - med) > 4, med, raw).astype(np.float32)
        e3 = gaussian_filter1d(edge, 6, axis=0, mode="nearest")
        glob = np.median(edge[W // 10:9 * W // 10], axis=0)
        sm = gaussian_filter1d(edge, 120, axis=0, mode="nearest")
        efar = (glob[None, :] + np.clip(sm - glob[None, :], -4, 4)).astype(np.float32)
        ramp = max(ease, 120)
    else:
        raw = np.median(a[:2] if where == "top" else a[-2:], axis=0).astype(np.float32)
        med = median_filter(raw, size=(61, 1), mode="nearest")
        edge = np.where(np.abs(raw - med) > 6, med, raw).astype(np.float32)
        e3 = gaussian_filter1d(edge, 3, axis=0, mode="nearest")
        glob = np.median(edge[W // 10:9 * W // 10], axis=0)
        sm = gaussian_filter1d(edge, 60, axis=0, mode="nearest")
        efar = (glob[None, :] + np.clip(sm - glob[None, :], -6, 6)).astype(np.float32)
        ramp = max(ease, 80)
    d = np.arange(1, n + 1, dtype=np.float32)
    w = np.clip(d / ramp, 0, 1)
    w = w * w * (3 - 2 * w)
    tone = (1 - w)[:, None, None] * e3[None] + w[:, None, None] * efar[None]
    g = calibrated(quilt(srcs, masks, n + 40, W, seed, th=th), tf, tm)[20:20 + n]
    rows = tone + g
    return rows[::-1] if where == "top" else rows


def content_box(g):
    """Printed-map box (top, bottom, left, right): outer neatline if found on all sides, else ink extent."""
    H, W = g.shape
    t, b, l, r = border_lines(g)
    if None not in (t, b, l, r) and b - t > 0.7 * H and r - l > 0.7 * W:
        return (t, b, l, r), "neatline"
    ink = g < (np.median(g) - 40)
    rows = np.where(ink.mean(1) > 0.002)[0]
    cols = np.where(ink.mean(0) > 0.002)[0]
    if len(rows) == 0 or len(cols) == 0:
        return (0, H - 1, 0, W - 1), "none"
    return (int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])), "ink"


def plan_pads(H, W, box, ratio, buffer):
    """Pads (t, b, l, r) that centre the printed map, reach the frame ratio, then add an equal buffer."""
    ct, cb, cl, cr = box
    mt, mb, ml, mr = ct, H - 1 - cb, cl, W - 1 - cr
    mv, mh = max(mt, mb), max(ml, mr)
    t, b, l, r = mv - mt, mv - mb, mh - ml, mh - mr
    H2, W2 = H + t + b, W + l + r
    if W2 / H2 < ratio:
        e = int(round(H2 * ratio)) - W2
        l, r, W2 = l + e // 2, r + e - e // 2, W2 + e
    else:
        e = int(round(W2 / ratio)) - H2
        t, b, H2 = t + e // 2, b + e - e // 2, H2 + e
    bx = int(round(buffer * W2))
    W3 = W2 + 2 * bx
    H3 = int(round(W3 / ratio))
    ey = H3 - H2
    by0, by1 = ey // 2, ey - ey // 2
    return (t + by0, b + by1, l + bx, r + bx), (W2, H2), (bx, by0)


def build_canvas(R, pads, srcs, masks, tf, tm, seed, ease, tone_mode="margin", th=24):
    """Add paper on all four sides; original pixels untouched."""
    H, W = R.shape[:2]
    t, b, l, r = pads
    A = np.concatenate([ext_rows(R, t, "top", srcs, masks, tf, tm, seed, ease, tone_mode, th), R.astype(np.float32),
                        ext_rows(R, b, "bottom", srcs, masks, tf, tm, seed + 100, ease, tone_mode, th)], 0)
    At = np.ascontiguousarray(A.transpose(1, 0, 2))
    st = [np.ascontiguousarray(s.transpose(1, 0, 2)) for s in srcs]
    mt_ = [np.ascontiguousarray(m.T) for m in masks]
    B = np.concatenate([ext_rows(At, l, "top", st, mt_, tf, tm, seed + 200, ease, tone_mode, th), At,
                        ext_rows(At, r, "bottom", st, mt_, tf, tm, seed + 300, ease, tone_mode, th)], 0).transpose(1, 0, 2)
    out = np.clip(np.round(B), 0, 255).astype(np.uint8)
    box = (t, t + H, l, l + W)
    assert np.array_equal(out[t:t + H, l:l + W], R), "original region changed"
    return out, box


def tex(p):
    f, m = band_split(p.astype(np.float32))
    return float(f.std()), float(m.std())


def seam_metrics(out, box, tf, tm):
    y0, y1, x0, x1 = box
    res = {"seam_diff_max": 0.0, "texture_ratio_fine": [], "texture_ratio_mid": []}
    strips = []
    Hh, Ww = out.shape[:2]
    if y0 > 0:
        res["seam_diff_max"] = max(res["seam_diff_max"], float(np.abs(out[y0].mean(0) - out[y0 - 1].mean(0)).max()))
        strips.append(out[max(0, y0 - 64):max(0, y0 - 4), Ww // 10:9 * Ww // 10])
    if y1 < Hh:
        res["seam_diff_max"] = max(res["seam_diff_max"], float(np.abs(out[y1].mean(0) - out[y1 - 1].mean(0)).max()))
        strips.append(out[y1 + 4:y1 + 64, Ww // 10:9 * Ww // 10])
    if x0 > 0:
        res["seam_diff_max"] = max(res["seam_diff_max"], float(np.abs(out[:, x0].mean(0) - out[:, x0 - 1].mean(0)).max()))
        strips.append(out[Hh // 10:9 * Hh // 10, max(0, x0 - 64):max(0, x0 - 4)])
    if x1 < Ww:
        res["seam_diff_max"] = max(res["seam_diff_max"], float(np.abs(out[:, x1].mean(0) - out[:, x1 - 1].mean(0)).max()))
        strips.append(out[Hh // 10:9 * Hh // 10, x1 + 4:x1 + 64])
    for s in strips:
        if s.shape[0] >= 12 and s.shape[1] >= 12:
            f, m = tex(s)
            res["texture_ratio_fine"].append(round(f / max(tf, 1e-6), 2))
            res["texture_ratio_mid"].append(round(m / max(tm, 1e-6), 2))
    res["seam_diff_max"] = round(res["seam_diff_max"], 2)
    return res


def tile(img, cy, cx, gain, h=220, w=360):
    yy = int(np.clip(cy - h // 2, 0, img.shape[0] - h))
    xx = int(np.clip(cx - w // 2, 0, img.shape[1] - w))
    c = img[yy:yy + h, xx:xx + w].astype(np.float32)
    if gain > 1:
        c = (c - np.median(c, (0, 1))) * gain + 128
    t = np.clip(c, 0, 255).astype(np.uint8)
    if gain > 1:
        t = cv2.rectangle(t.copy(), (0, 0), (w - 1, h - 1), (255, 0, 0), 2)
    return t


def qa_sheet(out, box, xf_out, win=None, cbox=None):
    """Seam crops (6x then normal), fold and corners, then a thumbnail with frame window (red) and map box (blue)."""
    H, W = out.shape[:2]
    y0, y1, x0, x1 = box
    seams = []
    if y0 > 0: seams.append((y0, W // 2))
    if y1 < H: seams.append((y1, W // 2))
    if x0 > 0: seams.append((H // 2, x0))
    if x1 < W: seams.append((H // 2, x1))
    for e in [(y0, int(W * .15)), (y1, int(W * .85)), (int(H * .15), x0), (int(H * .85), x1)]:
        if len(seams) >= 4: break
        seams.append(e)
    seams = seams[:4]
    tiles = [tile(out, cy, cx, 6) for cy, cx in seams]
    tiles += [tile(out, cy, cx, 1) for cy, cx in seams]
    fx = xf_out if xf_out else W // 2
    tiles += [tile(out, H // 2, fx, 6), tile(out, H // 4, fx, 1), tile(out, 110, 110, 1), tile(out, H - 110, W - 110, 1)]
    sheet = np.full((3 * 226 + 300, 4 * 366, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, 4)
        sheet[r * 226:r * 226 + 220, c * 366:c * 366 + 360] = t
    sc = 300 / H
    th = cv2.resize(out, (int(round(W * sc)), 300), interpolation=cv2.INTER_AREA)
    if win is not None:
        wx, wy, ww, wh = win
        th = cv2.rectangle(th.copy(), (int(wx * sc), int(wy * sc)), (int((wx + ww) * sc) - 1, int((wy + wh) * sc) - 1), (220, 0, 0), 1)
    if cbox is not None:
        ct, cb, cl, cr = cbox
        th = cv2.rectangle(th.copy(), (int(cl * sc), int(ct * sc)), (int(cr * sc), int(cb * sc)), (0, 60, 255), 1)
    th = th[:, :min(th.shape[1], sheet.shape[1])]
    sheet[3 * 226:3 * 226 + 300, :th.shape[1]] = th
    return sheet


def enc_png(a):
    return cv2.imencode(".png", cv2.cvtColor(a, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6])[1].tobytes()


def enc_jpg(a, q=86):
    return cv2.imencode(".jpg", cv2.cvtColor(a, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


def put(up, data, ctype):
    if len(data) > up.cap:
        return f"too_large:{len(data)}>{up.cap}"
    r = httpx.put(up.url, content=data, timeout=600,
                  headers={"Content-Type": ctype, "x-goog-content-length-range": f"0,{up.cap}"})
    return str(r.status_code)


class Upload(BaseModel):
    url: str
    cap: int


class SizeSpec(BaseModel):
    key: str
    ratio: float
    out_long_px: Optional[int] = None


class PipelineRequest(BaseModel):
    mode: Literal["assess", "build"] = "assess"
    file_name: Optional[str] = None
    url: Optional[str] = None
    expect_md5: Optional[str] = None
    expect_wh: Optional[List[int]] = None
    fold: bool = False
    dewarp: bool = False
    ease_px: int = 10
    tone_mode: Literal["margin", "edge", "smooth"] = "margin"
    grain_mode: Literal["margin", "rich", "donor"] = "margin"
    grain_file: Optional[str] = None
    grain_url: Optional[str] = None
    grain_scale: float = 1.0
    sizes: List[SizeSpec] = Field(default_factory=list)
    uploads: Dict[str, Upload] = Field(default_factory=dict)
    buffer: float = 0.02
    seed: int = 1
    auto_trim: bool = False
    trim_tblr: Optional[List[int]] = None
    deskew: Literal["auto", "off"] = "auto"
    deskew_min_deg: float = 0.03
    deskew_max_deg: float = 3.0
    deskew_max_spread_deg: float = 1.0
    max_work_long_px: int = 10000
    work_margin: float = 1.15


def pipeline(req: PipelineRequest) -> Dict[str, Any]:
    logger.info("Fetching map scan", source=req.file_name or req.url, mode=req.mode)
    if not (req.file_name or req.url):
        raise HTTPException(422, "file_name or url required")
    b, a = fetch(req.file_name, req.url)
    H, W = a.shape[:2]
    md5 = hashlib.md5(b).hexdigest()
    res = {"bytes": len(b), "md5": md5, "wh": [W, H]}
    if req.expect_md5 and req.expect_md5 != md5:
        res["gate_b"] = "FAIL md5 mismatch"
        return res
    if req.expect_wh and list(req.expect_wh) != [W, H]:
        res["gate_b"] = f"FAIL pixels {W}x{H} != log {req.expect_wh[0]}x{req.expect_wh[1]}"
        return res
    res["gate_b"] = "PASS"
    del b
    if req.trim_tblr or req.auto_trim:
        if req.trim_tblr:
            et = [max(0, int(v)) for v in req.trim_tblr]
            flags, pl = [], None
        else:
            et, flags, pl = auto_trim_edges(a)
        a = np.ascontiguousarray(a[et[0]:H - et[1], et[2]:W - et[3]])
        H, W = a.shape[:2]
        res["edge_trim_tblr"] = et
        res["trimmed_wh"] = [W, H]
        if pl is not None:
            res["edge_trim_paper_level"] = pl
        if flags:
            res["edge_trim_no_clean_edge"] = flags
        logger.info("Edge trim", trim=et, flags=flags)
    if req.deskew == "auto":
        na = neatline_angle(gray(a))
        if na is None:
            res["deskew"] = "no border found: scan left as is"
            res["border_fit"] = dict(neatline_angle.diag)
        else:
            ang, per, spread = na
            res["border_angle_deg"] = round(ang, 3)
            res["border_angle_per_side"] = per
            res["border_angle_spread_deg"] = round(spread, 3)
            res["border_fit"] = dict(neatline_angle.diag)
            if spread > req.deskew_max_spread_deg:
                res["deskew"] = "skipped: border sides disagree by over 1 degree (check preview)"
            elif abs(ang) > req.deskew_max_deg:
                res["deskew"] = "skipped: angle too large, check preview"
            elif abs(ang) < req.deskew_min_deg:
                res["deskew"] = "level already"
            else:
                a, crop = deskew(a, ang)
                H, W = a.shape[:2]
                after = neatline_angle(gray(a))
                res["deskew"] = "applied" if spread <= 0.35 else "applied (sides not quite parallel: levelled to their median, check the QA thumbnail)"
                res["deskew_crop_tblr"] = crop
                res["deskewed_wh"] = [W, H]
                res["border_angle_after_deg"] = None if after is None else round(after[0], 3)
                logger.info("Deskewed to border", angle=round(ang, 3), after=res["border_angle_after_deg"])
    if req.mode == "build" and max(W, H) > req.max_work_long_px and req.sizes and all(s.out_long_px for s in req.sizes):
        target = int(round(max(s.out_long_px for s in req.sizes) * req.work_margin))
        if target < max(W, H):
            k0 = target / max(W, H)
            a = cv2.resize(a, (int(round(W * k0)), int(round(H * k0))), interpolation=cv2.INTER_AREA)
            H, W = a.shape[:2]
            res["work_scale"] = round(k0, 4)
            res["work_wh"] = [W, H]
            logger.info("Downscaled for build", work_wh=[W, H])
    landscape = W >= H
    g = gray(a)
    borders = border_lines(g)
    xf, step, gutter = find_fold(g)
    colp = col_paper(g)
    if req.mode == "assess":
        dev = None
        if borders[0] is not None and borders[1] is not None:
            top = trace_line(g, borders[0], xf)
            bot = trace_line(g, borders[1], xf)
            dev = [round(float(np.ptp(top[150:-150])), 1), round(float(np.ptp(bot[150:-150])), 1)]
        L, S = max(W, H), min(W, H)
        sizes = {}
        for k, (lng, shrt) in {"8x10": (10, 8), "10x10": (10, 10), "11x14": (14, 11), "12x16": (16, 12), "16x20": (20, 16)}.items():
            r = lng / shrt
            ext = (L / r - S) / S if L / r >= S else (S * r - L) / L
            sizes[k] = {"dpi": int(min(L / lng, S / shrt)), "extension_pct": round(100 * ext, 1)}
        res.update({
            "orientation": "landscape" if landscape else "portrait",
            "borders_tblr": borders,
            "margins_tblr": [borders[0], H - 1 - borders[1] if borders[1] else None, borders[2], W - 1 - borders[3] if borders[3] else None],
            "fold_x": xf, "fold_step": round(step, 1), "gutter_darkening": round(gutter, 1),
            "border_ptp_top_bot": dev,
            "edge_tone_tblr": [int(np.median(g[:15])), int(np.median(g[-15:])), int(np.median(g[:, :15])), int(np.median(g[:, -15:]))],
            "paper_cols_40": [int(colp[i]) for i in range(0, W, max(1, W // 40))],
            "sizes": sizes,
            "recommend": {"fold": bool(step > 15 or gutter > 15), "dewarp": bool(dev is not None and max(dev) > 3)},
        })
        return res

    logger.info("Restoring", fold=req.fold, dewarp=req.dewarp)
    R = a.astype(np.float32)
    fold_range = None
    trim = [0, 0]
    if req.fold:
        R = flatten(a, xf)
    if req.dewarp:
        if borders[0] is None or borders[1] is None:
            raise HTTPException(422, "dewarp needs top and bottom border lines")
        R, tt, tb, _, _ = dewarp(R, borders[0], borders[1], xf)
        R = R[tt:R.shape[0] - tb]
        trim = [tt, tb]
    R = np.clip(np.round(R), 0, 255).astype(np.uint8)
    gR = gray(R)
    bR = border_lines(gR)
    if req.fold:
        cp = col_paper(gR)
        z = cp[max(0, xf - 150):xf + 150]
        base = cp[min(len(cp) - 300, xf + 400):min(len(cp), xf + 700)]
        fold_range = [int(np.ptp(z)), int(np.ptp(base)) if len(base) else None]
    line_ptp = None
    if req.dewarp and bR[0] is not None and bR[1] is not None:
        t2, b2 = trace_line(gR, bR[0], xf), trace_line(gR, bR[1], xf)
        line_ptp = [round(float(np.ptp(t2[150:-150])), 1), round(float(np.ptp(b2[150:-150])), 1)]
    qth = 24
    if req.grain_mode != "margin":
        srcs, masks, tf, tm = rich_grain(R, bR, req, res)
        qth = 48
    else:
        try:
            srcs, masks, tf, tm = grain_sources(R, margin_boxes(R.shape, bR))
        except HTTPException:
            srcs, masks, tf, tm = grain_sources(R, margin_boxes(R.shape, bR), adaptive=True)
            res["grain_fallback"] = "adaptive"
            logger.info("Grain sources via adaptive thresholds")
    res.update({"restored_wh": [R.shape[1], R.shape[0]], "trim_top_bot": trim, "fold_x": xf,
                "fold_zone_range_vs_baseline": fold_range, "border_ptp_after": line_ptp,
                "margin_texture_fine_mid": [round(tf, 2), round(tm, 2)], "files": {}})
    cbox, cmethod = content_box(gR)
    res["map_box_tblr"] = list(cbox)
    res["map_box_method"] = cmethod
    first = None
    for s in req.sizes:
        logger.info("Building print file", size=s.key)
        r = s.ratio if landscape else 1 / s.ratio
        pads, (W2, H2), (wx, wy) = plan_pads(R.shape[0], R.shape[1], cbox, r, req.buffer)
        out, box = build_canvas(R, pads, srcs, masks, tf, tm, req.seed + 10 * len(res["files"]), req.ease_px, req.tone_mode, qth)
        m = seam_metrics(out, box, tf, tm)
        ct, cb, cl, cr = cbox[0] + box[0], cbox[1] + box[0], cbox[2] + box[2], cbox[3] + box[2]
        mf = [ct - wy, (wy + H2 - 1) - cb, cl - wx, (wx + W2 - 1) - cr]
        off = [round((mf[2] - mf[3]) / 2, 1), round((mf[0] - mf[1]) / 2, 1)]
        if first is None:
            first = (out, box, (wx, wy, W2, H2), (ct, cb, cl, cr))
        fin, k = out, 1.0
        if s.out_long_px and s.out_long_px < max(W2, H2):
            k = s.out_long_px / max(W2, H2)
            fin = cv2.resize(out, (int(round(out.shape[1] * k)), int(round(out.shape[0] * k))), interpolation=cv2.INTER_AREA)
        data = enc_png(fin)
        info = {"wh": [fin.shape[1], fin.shape[0]], "bytes": len(data), "md5": hashlib.md5(data).hexdigest(),
                "frame_window_px": [round(wx * k), round(wy * k), round(W2 * k), round(H2 * k)],
                "region_fraction_scale": round(out.shape[1] / W2, 4),
                "map_margins_in_frame_tblr": mf, "centre_offset_xy_px": off,
                "added_px_tblr": list(pads), **m}
        tex_ok = all(0.8 <= v <= 1.25 for v in m["texture_ratio_fine"] + m["texture_ratio_mid"])
        info["numeric_pass"] = bool(m["seam_diff_max"] <= 1.5 and tex_ok and max(abs(off[0]), abs(off[1])) <= 1)
        if s.key in req.uploads:
            info["upload"] = put(req.uploads[s.key], data, "image/png")
        res["files"][s.key] = info
    if first is not None:
        out, box, win, cb_ = first
        sheet = qa_sheet(out, box, (xf + box[2]) if req.fold else None, win, cb_)
        qd = enc_jpg(sheet)
        res["qa_bytes"] = len(qd)
        if "qa" in req.uploads:
            res["qa_upload"] = put(req.uploads["qa"], qd, "image/jpeg")
    logger.info("pipeline done", md5=md5)
    return res


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with open(src) as f:
        raw = json.load(f)
    reqs = raw if isinstance(raw, list) else [raw]
    out = []
    for r in reqs:
        t0 = time.time()
        try:
            res = pipeline(PipelineRequest(**r))
            res["ok"] = True
        except PipelineError as e:
            res = {"ok": False, "status": e.status, "error": e.detail}
        except Exception as e:  # noqa: BLE001
            res = {"ok": False, "status": 500, "error": f"{type(e).__name__}: {e}",
                   "trace": traceback.format_exc()[-1500:]}
        res["file_name"] = r.get("file_name") or r.get("url")
        res["seconds"] = round(time.time() - t0, 1)
        out.append(res)
    with open(dst, "w") as f:
        json.dump(out if isinstance(raw, list) else out[0], f, separators=(",", ":"))


if __name__ == "__main__":
    main()
