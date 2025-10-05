# ecg_digitizer_advanced.py
import cv2
import numpy as np

# ---------- helpers ----------
def _to_gray(img):
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def _auto_contrast(gray):
    # CLAHE helps a lot on faint grids/curves
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def _estimate_grid_period(gray):
    """
    Estimate grid period (pixels between vertical/horizontal minor gridlines)
    using projection + autocorrelation. Returns (px_x, px_y).
    """
    # Normalize
    g = cv2.GaussianBlur(gray, (5,5), 0)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)

    # Emphasize gridlines (high frequency vertical/horizontal)
    # Try a simple tophat (morphological) approach:
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    tophat_v = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, kernel_v)
    tophat_h = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, kernel_h)

    proj_x = np.mean(tophat_v, axis=0)
    proj_y = np.mean(tophat_h, axis=1)

    def _period_from_autocorr(x):
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)
        ac = np.correlate(x, x, mode='full')[len(x)-1:]
        ac /= (np.max(ac) + 1e-8)
        # find first local max after lag>5 px
        lag_min = 5
        peaks = []
        for i in range(lag_min+1, min(len(ac)-1, 400)):
            if ac[i] > ac[i-1] and ac[i] > ac[i+1]:
                peaks.append((ac[i], i))
        if not peaks:
            return None
        peaks.sort(reverse=True)
        return peaks[0][1]

    px_x = _period_from_autocorr(proj_x) or 10
    px_y = _period_from_autocorr(proj_y) or 10
    return int(px_x), int(px_y)

def _find_panels(gray):
    """
    Find rectangular panels in a 12-lead page (each lead has a framed box).
    Return list of bounding boxes (x,y,w,h) sorted by reading order.
    """
    # Edge + contour detection
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(blur, 30, 80)
    # close gaps
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    edges = cv2.dilate(edges, k, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    h, w = gray.shape
    for c in contours:
        x,y,wc,hc = cv2.boundingRect(c)
        area = wc*hc
        # filter too small / too large
        if area < 0.01*h*w or area > 0.9*h*w:
            continue
        ar = wc/float(hc)
        # heuristic: ECG panels are wide rectangles
        if ar > 2.5 and 0.1*h < hc < 0.5*h:
            boxes.append((x,y,wc,hc))
    # sort by top-left reading order
    boxes.sort(key=lambda b:(b[1]//30, b[0]))
    return boxes

def _pick_lead_II_box(boxes):
    """
    Heuristic: In many layouts, Lead II is the second panel in the left column (or second overall).
    If you know your sheet layout, adjust the index rule below.
    """
    if not boxes:
        return None
    # Try the 2nd or 3rd box as Lead II candidate; fallback to the largest
    candidates = []
    if len(boxes) >= 2:
        candidates.append(boxes[1])
    if len(boxes) >= 3:
        candidates.append(boxes[2])
    if not candidates:
        candidates = boxes
    # pick tallest as Lead II (tends to show rhythm strip)
    lead = max(candidates, key=lambda b:b[3])
    return lead

def _remove_text(gray_panel):
    """
    Remove text/annotations via morphological opening and inpainting.
    """
    th = cv2.adaptiveThreshold(gray_panel,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,35,5)
    # Assume text is thin; detect with small kernels
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    text_mask = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    # inpaint to clean
    cleaned = cv2.inpaint(gray_panel, text_mask, 3, cv2.INPAINT_TELEA)
    return cleaned

def _isolate_curve(gray_panel):
    """
    Enhance curve strokes, remove gridlines using directional morphology,
    then skeletonize to one-pixel centerline.
    """
    # Equalize for contrast
    g = _auto_contrast(gray_panel)

    # Remove vertical & horizontal gridlines
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    no_v = cv2.morphologyEx(g, cv2.MORPH_OPEN, kv)
    no_h = cv2.morphologyEx(g, cv2.MORPH_OPEN, kh)
    grid = cv2.max(no_v, no_h)
    degrid = cv2.subtract(g, grid)

    # Threshold to binary strokes
    _, bw = cv2.threshold(degrid, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Thin to 1-px skeleton
    ske = cv2.ximgproc.thinning(bw) if hasattr(cv2, "ximgproc") else _zhang_suen_thinning(bw)
    return ske

def _zhang_suen_thinning(bw):
    """Fallback pure-python thinning if ximgproc missing."""
    img = (bw>0).astype(np.uint8)
    prev = np.zeros_like(img)
    while True:
        diff = np.any(img != prev)
        prev = img.copy()
        # step 1
        m = np.zeros_like(img)
        for i in range(1, img.shape[0]-1):
            for j in range(1, img.shape[1]-1):
                P = img[i-1:i+2, j-1:j+2]
                p = P.flatten()
                p2,p3,p4,p5,p6,p7,p8,p9 = p[1],p[2],p[5],p[8],p[7],p[6],p[3],p[0]
                if img[i,j]==1:
                    neighbors = np.sum(p) - 1
                    transitions = ((p2==0 and p3==1) + (p3==0 and p4==1) + (p4==0 and p5==1) +
                                   (p5==0 and p6==1) + (p6==0 and p7==1) + (p7==0 and p8==1) +
                                   (p8==0 and p9==1) + (p9==0 and p2==1))
                    if 2 <= neighbors <= 6 and transitions == 1 and (p2*p4*p6==0) and (p4*p6*p8==0):
                        m[i,j] = 1
        img[m==1] = 0
        # step 2
        m[:] = 0
        for i in range(1, img.shape[0]-1):
            for j in range(1, img.shape[1]-1):
                P = img[i-1:i+2, j-1:j+2]
                p = P.flatten()
                p2,p3,p4,p5,p6,p7,p8,p9 = p[1],p[2],p[5],p[8],p[7],p[6],p[3],p[0]
                if img[i,j]==1:
                    neighbors = np.sum(p) - 1
                    transitions = ((p2==0 and p3==1) + (p3==0 and p4==1) + (p4==0 and p5==1) +
                                   (p5==0 and p6==1) + (p6==0 and p7==1) + (p7==0 and p8==1) +
                                   (p8==0 and p9==1) + (p9==0 and p2==1))
                    if 2 <= neighbors <= 6 and transitions == 1 and (p2*p4*p8==0) and (p2*p6*p8==0):
                        m[i,j] = 1
        img[m==1] = 0
        if not diff:
            break
    return (img*255).astype(np.uint8)

def _vectorize_centerline(skeleton):
    """
    For each x-column, take median y of white pixels; interpolate gaps.
    Returns y array (pixels from top).
    """
    h, w = skeleton.shape
    ys = np.full(w, np.nan, dtype=np.float32)
    ys_list = np.where(skeleton>0)
    # Build per-column lists
    cols = {}
    for r,c in zip(*ys_list):
        cols.setdefault(c, []).append(r)
    for c, rows in cols.items():
        ys[c] = np.median(rows)

    # interpolate missing
    idx = np.arange(w)
    mask = ~np.isnan(ys)
    if mask.sum() < 10:
        return None
    ys = np.interp(idx, idx[mask], ys[mask])
    return ys

def _calibrate_to_signal(ys_pixels, px_x, px_y, fs=360, time_mm_per_sec=25.0, mm_per_mV=10.0):
    """
    Convert pixel y to mV, resample to fs Hz using measured grid periods.
    px_x: pixels per minor grid (time)
    px_y: pixels per minor grid (amplitude)
    """
    # pixels -> mm
    mm_y = ys_pixels / (px_y + 1e-6)  # pixels / (pixels per mm-grid unit)
    # mm -> mV (10 mm = 1 mV)
    mv = mm_y / mm_per_mV
    # invert baseline (graphic y-down -> signal up)
    mv = -(mv - np.nanmedian(mv))

    # time sampling: each minor grid = 1 mm; at 25 mm/s -> 1 mm = 0.04 s
    # so px_x pixels -> 0.04 s; sample period per pixel:
    sec_per_pixel = 0.04 / (px_x + 1e-6)
    # resample to fs
    n = len(mv)
    t = np.arange(n) * sec_per_pixel
    t_new = np.arange(0, t[-1], 1.0/fs) if n>1 else np.array([0])
    if len(t_new) < 5:
        return None
    mv_resampled = np.interp(t_new, t, mv)
    return mv_resampled

# ---------- public API ----------
def extract_leadII_signal_with_overlay(img_bgr, fs=360, target_len=1800):
    """
    Returns (signal_np, overlay_bgr, debug_dict)
    signal_np is resampled @ fs; overlay shows chosen panel + skeleton.
    """
    gray = _to_gray(img_bgr)
    gray = _auto_contrast(gray)

    # find panels
    boxes = _find_panels(gray)
    if not boxes:
        return None, img_bgr, {"reason":"no_panels"}
    lead_box = _pick_lead_II_box(boxes)
    if lead_box is None:
        return None, img_bgr, {"reason":"lead_pick_failed"}

    x,y,w,h = lead_box
    panel = gray[y:y+h, x:x+w]
    panel_rgb = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)

    # remove text/annotations
    panel_clean = _remove_text(panel)

    # estimate grid periods
    px_x, px_y = _estimate_grid_period(panel_clean)

    # isolate curve & skeletonize
    skeleton = _isolate_curve(panel_clean)
    ys = _vectorize_centerline(skeleton)
    if ys is None:
        return None, panel_rgb, {"reason":"vectorize_failed"}

    # overlay for debug
    overlay = panel_rgb.copy()
    # draw skeleton points
    ys_int = np.clip(ys.astype(int), 0, skeleton.shape[0]-1)
    for c in range(0, skeleton.shape[1], max(1, skeleton.shape[1]//800)):
        overlay[ys_int[c], c] = (0,0,255)
    cv2.rectangle(img_bgr, (x,y), (x+w, y+h), (0,255,0), 2)

    # calibrate & resample
    mv = _calibrate_to_signal(ys, px_x=px_x, px_y=px_y, fs=fs)
    if mv is None:
        return None, overlay, {"reason":"resample_failed"}

    # if you want a fixed window length (e.g., first 5s)
    T = int(fs*5)
    if len(mv) >= T:
        mv = mv[:T]
    else:
        pad = np.zeros(T, dtype=np.float32)
        pad[:len(mv)] = mv
        mv = pad

    return mv.astype(np.float32), overlay, {
        "grid_px": (px_x, px_y),
        "panel_box": (x,y,w,h)
    }
