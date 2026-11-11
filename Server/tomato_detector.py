#!/usr/bin/env python3
# File name   : tomato_detector.py
# Description : Detekcija cheri paradajza na slici sa kamere robota
# Author      : napisano za master rad Adnana Malagica (dopuna Adeept PiCar-Pro softvera)
#
# KAKO RADI (4 koraka po frejmu):
#   1) MASKA BOJE  - piksel je "crven" ako mu je nijansa (H) u opsegu oko
#      crvene I zasicenost (S) i osvetljenost (V) iznad praga. Opseg nijanse
#      se "obavija" oko 0/180 (crvena je u OpenCV HSV-u na OBA kraja skale:
#      0..~15 i ~165..180) - stari kod to nije radio, pa je deo crvene boje
#      paradajza uvek otpadao, a prag zasicenosti je bio toliko nizak
#      (S>=17..50) da su i koza, roze sare na tepihu itd. ulazile u masku.
#   2) KANDIDATI   - svaka mrlja iz maske postaje kandidat ako prodje
#      jeftine (namerno popustljive) provere velicine i oblika.
#   3) ML OCENA    - za svakog kandidata se racuna ~80 osobina (oblik,
#      boja iznutra i oko mrlje, odsjaj, tekstura, histogrami, gradijenti) i
#      mala neuronska mreza (MLP, istrenirana van robota - vidi
#      tomato_model.json) daje verovatnocu 0..1 da je to cheri paradajz.
#      Mreza se izvrsava cistim numpy-jem - na Raspberry Pi-ju NIJE potrebna
#      nijedna dodatna biblioteka.
#   4) PRACENJE    - odluka se NE donosi na osnovu jednog frejma. Tracker
#      prati metu kroz vreme (povezuje kandidate po poziciji/velicini),
#      "potvrdi" je tek posle nekoliko uzastopnih dobrih frejmova i "pusti"
#      je tek kad je nema duze vreme (histereza). Ovo uklanja treperenje
#      "vidim / ne vidim" zbog kog je robot stalno prelazio iz
#      CENTRIRANJE nazad u TRAZENJE.
#
# Modul ne zavisi od hardvera (servo, motori, kamera) - moze se testirati
# na obicnom racunaru nad slikama (vidi test_tomato_detector.py).

import json
import math
import os
import threading
import time

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_THIS_DIR, 'tomato_model.json')

# ---------------------------------------------------------------------------
# Podrazumevana podesavanja detekcije (mogu se promeniti u tomato_config.json)
# ---------------------------------------------------------------------------
DETECTOR_DEFAULTS = {
    # obrada maske na manjoj rezoluciji (brze + manje suma); osobine se
    # racunaju na izrezu iz PUNE rezolucije
    'proc_width': 320,
    # crvena: centar nijanse (OpenCV H je 0..179) i poluopseg; opseg se
    # obavija oko 0/180
    'hue_center': 2,
    'hue_halfwidth': 15,
    's_min': 90,
    # dodatni nivoi praga zasicenosti (s_min + s_step, s_min + s_step2) - vidi
    # find_candidates(). Treci nivo odvaja paradajz od JAKO rumene koze
    # (prsti na kameri robota imaju S do ~130, paradajz ~180-230).
    's_step': 45,
    's_step2': 85,
    'v_min': 40,
    # pre-filter kandidata (namerno popustljiv - konacnu odluku donosi ML)
    'min_area_px': 30,          # na proc rezoluciji (320 sirine)
    'max_area_frac': 0.40,
    'min_circularity': 0.35,
    'min_solidity': 0.70,
    'min_aspect': 0.40,
    'max_candidates': 8,
    # prag verovatnoce da kandidat uopste udje u tracker
    'p_candidate': 0.30,
    # prag za POKRETANJE novog traga (mora biti uverljiviji kandidat)
    'p_new_track': 0.55,
    # tracker: potvrda / gubitak (histereza)
    'confirm_conf': 0.60,
    'confirm_hits': 3,
    'drop_conf': 0.25,
    'max_unseen_s': 0.8,
    'ema_conf': 0.40,
    'ema_pos': 0.55,
}

FEAT_CROP = 32


def _circ_hue_offset(h, center):
    """Razlika nijanse od centra u stepenima OpenCV skale (-90..+90),
    ispravno preko granice 0/180."""
    return ((h.astype(np.int16) - int(center) + 90) % 180) - 90


def red_mask(bgr_small, cfg, s_min=None, hsv=None):
    """Binarna maska 'crvenih' piksela, sa obavijanjem nijanse oko 0/180."""
    if hsv is None:
        hsv = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HSV)
    hc = int(cfg['hue_center']) % 180
    hw = int(cfg['hue_halfwidth'])
    s_min = int(cfg['s_min'] if s_min is None else s_min)
    v_min = int(cfg['v_min'])
    lo = hc - hw
    hi = hc + hw
    if lo < 0:
        m1 = cv2.inRange(hsv, np.array([0, s_min, v_min]), np.array([hi, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([180 + lo, s_min, v_min]), np.array([179, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)
    elif hi > 179:
        m1 = cv2.inRange(hsv, np.array([lo, s_min, v_min]), np.array([179, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([0, s_min, v_min]), np.array([hi - 180, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)
    else:
        mask = cv2.inRange(hsv, np.array([lo, s_min, v_min]), np.array([hi, 255, 255]))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
    # zatvaranje popunjava "rupu" od belog odsjaja na paradajzu
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
    return mask


class Candidate(object):
    __slots__ = ('contour', 'x', 'y', 'r', 'bbox', 'area', 'circularity',
                 'solidity', 'aspect', 'features', 'prob')

    def as_dict(self):
        return {'x': round(float(self.x), 1), 'y': round(float(self.y), 1),
                'r': round(float(self.r), 1), 'p': round(float(self.prob), 3),
                'circ': round(float(self.circularity), 2)}


def _contours_to_cands(cnts, scale, cfg, sw, sh):
    max_area = cfg['max_area_frac'] * sw * sh
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < cfg['min_area_px'] or area > max_area:
            continue
        per = cv2.arcLength(c, True)
        if per <= 0:
            continue
        circ = 4.0 * math.pi * area / (per * per)
        if circ < cfg['min_circularity']:
            continue
        hull = cv2.convexHull(c)
        harea = cv2.contourArea(hull)
        solidity = area / harea if harea > 0 else 0.0
        if solidity < cfg['min_solidity']:
            continue
        (rw, rh) = cv2.minAreaRect(c)[1]
        aspect = min(rw, rh) / max(rw, rh) if max(rw, rh) > 0 else 0.0
        if aspect < cfg['min_aspect']:
            continue
        cand = Candidate()
        cand.contour = (c.astype(np.float32) * scale)
        (cx, cy), r = cv2.minEnclosingCircle(cand.contour)
        cand.x, cand.y, cand.r = cx, cy, r
        x, y, w, h = cv2.boundingRect(c)
        cand.bbox = (int(x * scale), int(y * scale), int(math.ceil(w * scale)), int(math.ceil(h * scale)))
        cand.area = area * scale * scale
        cand.circularity = circ
        cand.solidity = solidity
        cand.aspect = aspect
        cand.features = None
        cand.prob = 0.0
        out.append(cand)
    return out


def _same_object(a, b):
    d = math.hypot(a.x - b.x, a.y - b.y)
    ratio = a.r / max(b.r, 1e-3)
    return d < 0.5 * max(a.r, b.r) and 0.6 < ratio < 1.65


def find_candidates(bgr, cfg, mask_small=None):
    """Vraca (lista Candidate u koordinatama PUNE slike, maska_small).

    Maska se pravi na VISE nivoa zasicenosti (s_min, s_min+45, s_min+85). Kada
    paradajz dodiruje nesto slicne boje (dlan, roze sara tepiha), na nizem
    pragu se sve spoji u jednu nepravilnu mrlju koja ne prolazi proveru
    oblika - ali na visem pragu ostane samo jako zasiceno "jezgro"
    paradajza kao zasebna okrugla mrlja. Duplikati (isti objekat na oba
    nivoa) se spajaju."""
    H, W = bgr.shape[:2]
    pw = int(cfg['proc_width'])
    if W > pw:
        scale = W / float(pw)
        small = cv2.resize(bgr, (pw, int(round(H / scale))), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        small = bgr
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    base = int(cfg['s_min'])
    levels = [base]
    for step in (int(cfg.get('s_step', 45)), int(cfg.get('s_step2', 85))):
        lv = min(base + step, 200)
        if step > 0 and lv > levels[-1] + 20:
            levels.append(lv)
    cands = []
    first_mask = None
    for lv in levels:
        m = red_mask(small, cfg, s_min=lv, hsv=hsv) if (mask_small is None or lv != base) else mask_small
        if first_mask is None:
            first_mask = m
        sh, sw = m.shape[:2]
        cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        for c in _contours_to_cands(cnts, scale, cfg, sw, sh):
            dup = None
            for k in cands:
                if _same_object(c, k):
                    dup = k
                    break
            if dup is None:
                cands.append(c)
            elif c.circularity * c.solidity > dup.circularity * dup.solidity:
                cands[cands.index(dup)] = c
    # zadrzi najvece (ogranicava CPU na Pi-ju)
    cands.sort(key=lambda k: k.area, reverse=True)
    return cands[:int(cfg['max_candidates'])], first_mask


def _hog(gray, cells=2, bins=8):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = ang % 180.0
    n = gray.shape[0]
    cs = n // cells
    b = np.minimum((ang / (180.0 / bins)).astype(np.int32), bins - 1)
    out = np.zeros((cells, cells, bins), np.float32)
    for cy in range(cells):
        for cx in range(cells):
            mm = mag[cy * cs:(cy + 1) * cs, cx * cs:(cx + 1) * cs].ravel()
            bb = b[cy * cs:(cy + 1) * cs, cx * cs:(cx + 1) * cs].ravel()
            out[cy, cx] = np.bincount(bb, weights=mm, minlength=bins)[:bins]
    out = out.ravel()
    nrm = np.linalg.norm(out)
    if nrm > 0:
        out /= nrm
    return out, mag


def candidate_features(bgr, cand):
    """Vektor osobina (float32) za jednog kandidata. MORA ostati identican
    onome sto je korisceno pri treniranju modela (tomato_model.json)."""
    H, W = bgr.shape[:2]
    x, y, w, h = cand.bbox
    pad = max(4, int(0.25 * max(w, h)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    crop = bgr[y0:y1, x0:x1]
    cm = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cnt = (cand.contour - np.array([x0, y0], np.float32)).astype(np.int32)
    cv2.drawContours(cm, [cnt], -1, 255, -1)
    S = FEAT_CROP
    crop_s = cv2.resize(crop, (S, S), interpolation=cv2.INTER_AREA)
    m = cv2.resize(cm, (S, S), interpolation=cv2.INTER_NEAREST) > 127
    ring = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool) & ~m
    if m.sum() < 4:
        m = np.zeros((S, S), bool)
        m[S // 4:3 * S // 4, S // 4:3 * S // 4] = True
    if ring.sum() < 4:
        ring = ~m

    hsv = cv2.cvtColor(crop_s, cv2.COLOR_BGR2HSV)
    hh = _circ_hue_offset(hsv[:, :, 0], 0).astype(np.float32)
    ss = hsv[:, :, 1].astype(np.float32)
    vv = hsv[:, :, 2].astype(np.float32)
    hi, si, vi = hh[m], ss[m], vv[m]
    ho, so, vo = hh[ring], ss[ring], vv[ring]

    f = []
    # --- oblik ---
    f.append(math.log(max(cand.area, 1.0) / float(W * H)))
    f.append(min(cand.circularity, 1.2))
    f.append(cand.solidity)
    f.append(cand.aspect)
    f.append(cand.area / (w * h) if w * h > 0 else 0.0)
    f.append(cand.area / (math.pi * cand.r * cand.r) if cand.r > 0 else 0.0)
    f.append(1.0 if (x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1) else 0.0)
    # --- boja unutar mrlje ---
    f += [float(np.mean(hi)) / 30.0, float(np.std(hi)) / 30.0]
    f += [float(np.mean(si)) / 255.0, float(np.std(si)) / 255.0, float(np.percentile(si, 20)) / 255.0]
    f += [float(np.mean(vi)) / 255.0, float(np.std(vi)) / 255.0,
          float(np.percentile(vi, 90)) / 255.0, float(np.percentile(vi, 10)) / 255.0]
    f.append(float(np.mean(si > 150)))
    # odsjaj (beli sjaj na glatkoj kozici paradajza)
    vmax = float(vv[m | ring].max()) if (m | ring).any() else 0.0
    spec = m & (vv > 0.85 * vmax) & (ss < 0.6 * max(float(np.mean(si)), 1.0))
    f.append(float(spec.sum()) / float(m.sum()))
    # --- kontrast u odnosu na okolinu ---
    f += [float(np.mean(so)) / 255.0, float(np.mean(vo)) / 255.0, float(np.mean(np.abs(ho))) / 90.0]
    f += [(float(np.mean(si)) - float(np.mean(so))) / 255.0, (float(np.mean(vi)) - float(np.mean(vo))) / 255.0]
    # --- tekstura (jagoda/malina/koza su "hrapave", paradajz je gladak) ---
    gray = cv2.cvtColor(crop_s, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    hog, mag = _hog(gray)
    vmean = float(np.mean(vi)) + 1.0
    f.append(float(np.mean(lap[m])) / vmean)
    f.append(float(np.mean(mag[m])) / vmean)
    # --- radijalni profil (loptasto sencenje) ---
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    ys, xs = np.nonzero(m)
    cyy, cxx = float(ys.mean()), float(xs.mean())
    rr = np.sqrt((yy - cyy) ** 2 + (xx - cxx) ** 2)
    rmax = float(rr[m].max()) + 1e-3
    for lo, hi_ in ((0.0, 0.4), (0.4, 0.75), (0.75, 1.01)):
        z = m & (rr >= lo * rmax) & (rr < hi_ * rmax)
        if z.any():
            f.append(float(np.mean(vv[z])) / vmean)
            f.append(float(np.mean(ss[z])) / 255.0)
        else:
            f += [1.0, 0.0]
    # --- histogram nijansa x zasicenost unutar mrlje (6 x 3) ---
    hb = np.clip(((hi + 45.0) / 15.0).astype(np.int32), 0, 5)
    sb = np.digitize(si, [100.0, 170.0])
    hs = np.bincount(hb * 3 + sb, minlength=18)[:18].astype(np.float32)
    hs /= max(float(hs.sum()), 1.0)
    f += hs.tolist()
    # --- gradijenti (obris) ---
    f += hog.tolist()
    return np.asarray(f, np.float32)


FEAT_LEN = 7 + 2 + 3 + 4 + 1 + 1 + 3 + 2 + 2 + 6 + 18 + 32


class TomatoModel(object):
    """Ansambl malih neuronskih mreza (MLP) + logisticka regresija, sacuvan
    kao JSON; unapred proracun u cistom numpy-ju. Konacna verovatnoca je
    TEZINSKI PROSEK clanova ansambla - jedan "presamouveren" clan tako ne
    moze sam da odbaci pravi paradajz (upravo to se desavalo sa starim
    modelom). Logisticka regresija ima vecu tezinu jer se na realnim
    slikama sa kamere robota pokazala najstabilnijom."""

    def __init__(self, path=MODEL_PATH):
        self.ready = False
        self.info = ''
        self.members = []
        self.weights = []
        try:
            with open(path, 'r') as fh:
                d = json.load(fh)
            self.mean = np.asarray(d['mean'], np.float32)
            self.scale = np.asarray(d['scale'], np.float32)
            members = d.get('members') or [{'layers': d['layers']}]
            for m in members:
                self.members.append([(np.asarray(L['W'], np.float32), np.asarray(L['b'], np.float32))
                                     for L in m['layers']])
                self.weights.append(float(m.get('weight', 1.0)))
            wsum = sum(self.weights)
            self.weights = [w / wsum for w in self.weights]
            if self.mean.shape[0] != FEAT_LEN:
                raise ValueError('duzina osobina: model=%d kod=%d' % (self.mean.shape[0], FEAT_LEN))
            self.info = d.get('info', '')
            self.ready = True
            print('[TOMATO] Model učitan: %d clan(a) ansambla (%s)' % (len(self.members), self.info), flush=True)
        except Exception as e:
            print('[TOMATO] UPOZORENJE: model nije učitan (%s) - koristim '
                  'heuristicku ocenu oblika/boje.' % e, flush=True)

    @staticmethod
    def _forward(layers, X):
        a = X
        for i, (Wm, b) in enumerate(layers):
            a = a.dot(Wm) + b
            if i < len(layers) - 1:
                a = np.maximum(a, 0.0)
        z = np.clip(a[:, 0], -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, X):
        X = (np.asarray(X, np.float32) - self.mean) / self.scale
        X = np.clip(X, -8.0, 8.0)   # zastita od ekstremnih (neocekivanih) vrednosti
        return np.sum([w * self._forward(L, X) for w, L in zip(self.weights, self.members)], axis=0)


def heuristic_prob(c, feats):
    """Rezervna ocena ako model nije ucitan: okruglo + puno + jako zasiceno."""
    sat = feats[9]  # srednja S unutar (0..1)
    s = (min(c.circularity, 1.0) - 0.5) * 2.0 + (c.solidity - 0.8) * 3.0 + (sat - 0.45) * 3.0
    return float(1.0 / (1.0 + math.exp(-2.0 * s)))


class Track(object):
    def __init__(self, c, t):
        self.x, self.y, self.r = c.x, c.y, c.r
        self.conf = c.prob * 0.8
        self.hits = 1
        self.misses = 0
        self.last_seen = t
        self.confirmed = False
        self.prob = c.prob


class TomatoTracker(object):
    """Prati JEDNU metu kroz vreme, sa histerezom (vidi opis na vrhu)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.track = None

    def reset(self):
        self.track = None

    def update(self, cands, t):
        cfg = self.cfg
        good = [c for c in cands if c.prob >= cfg['p_candidate']]
        tr = self.track
        if tr is not None:
            best, best_s = None, -1e9
            gate = max(2.5 * tr.r, 40.0)
            for c in good:
                d = math.hypot(c.x - tr.x, c.y - tr.y)
                ratio = c.r / max(tr.r, 1.0)
                if d < gate and 0.45 < ratio < 2.2:
                    sc = c.prob - 0.3 * d / gate   # najverovatniji, pa najblizi
                    if sc > best_s:
                        best, best_s = c, sc
            # ako se negde drugde pojavi MNOGO uverljiviji kandidat, a trenutna
            # meta je slaba (npr. zakacila se za prste), predji na njega
            others = [c for c in good if c is not best and c.prob >= cfg['p_new_track']]
            if others:
                o = max(others, key=lambda k: k.prob)
                cur = best.prob if best is not None else 0.0
                if o.prob > cur + 0.3 and tr.conf < 0.85:
                    self.track = Track(o, t)
                    return self.track
            if best is not None:
                a = cfg['ema_pos']
                tr.x = a * best.x + (1 - a) * tr.x
                tr.y = a * best.y + (1 - a) * tr.y
                tr.r = a * best.r + (1 - a) * tr.r
                tr.conf = (1 - cfg['ema_conf']) * tr.conf + cfg['ema_conf'] * best.prob
                tr.prob = best.prob
                tr.hits += 1
                tr.misses = 0
                tr.last_seen = t
            else:
                tr.conf *= 0.75
                tr.misses += 1
            if not tr.confirmed and tr.conf >= cfg['confirm_conf'] and tr.hits >= cfg['confirm_hits']:
                tr.confirmed = True
            if tr.conf < cfg['drop_conf'] or (t - tr.last_seen) > cfg['max_unseen_s']:
                self.track = None
                tr = None
        if self.track is None:
            starters = [c for c in good if c.prob >= cfg['p_new_track']]
            if starters:
                c = max(starters, key=lambda k: k.prob * (1.0 + 0.002 * k.r))
                self.track = Track(c, t)
        return self.track


class TomatoDetector(object):
    """Kompletan lanac: maska -> kandidati -> ML -> tracker. Thread-safe citanje rezultata."""

    def __init__(self, cfg=None, model_path=MODEL_PATH):
        self.cfg = dict(DETECTOR_DEFAULTS)
        if cfg:
            self.cfg.update({k: v for k, v in cfg.items() if k in DETECTOR_DEFAULTS})
        self._sanitize()
        self.model = TomatoModel(model_path)
        self.tracker = TomatoTracker(self.cfg)
        self._lock = threading.Lock()
        self._result = None
        self.last_mask = None
        self.frame_w = 640
        self.frame_h = 480
        self.fps = 0.0
        self._t_prev = None
        # IZMENA: oblasti kadra u kojima se paradajz IGNORISE (npr. kantica sa
        # vec ubranim paradajzom - robot ne sme da ih "bere" ponovo).
        # Lista (x0, y0, x1, y1) u pikselima; postavlja je CV nit (camera_opencv.py).
        self.exclude_boxes = []

    def _sanitize(self):
        """Zastita od lose snimljene kalibracije (npr. tap na kozu/tepih):
        boja mora biti blizu crvene i dovoljno zasicena, inace podrazumevano."""
        c = self.cfg
        red_off = abs(((int(c['hue_center']) + 90) % 180) - 90)
        if red_off > 25 or not (6 <= int(c['hue_halfwidth']) <= 30):
            for k in ('hue_center', 'hue_halfwidth'):
                c[k] = DETECTOR_DEFAULTS[k]
        c['s_min'] = int(max(80, min(200, c['s_min'])))
        c['v_min'] = int(max(20, min(160, c['v_min'])))

    def set_color(self, hue_center=None, hue_halfwidth=None, s_min=None, v_min=None):
        if hue_center is not None:
            self.cfg['hue_center'] = int(hue_center) % 180
        if hue_halfwidth is not None:
            self.cfg['hue_halfwidth'] = int(max(6, min(30, hue_halfwidth)))
        if s_min is not None:
            # nikad ispod 80 - inace koza/roze tepih ulaze u masku
            self.cfg['s_min'] = int(max(80, min(200, s_min)))
        if v_min is not None:
            self.cfg['v_min'] = int(max(20, min(160, v_min)))
        self._sanitize()

    def score(self, bgr, cands):
        if not cands:
            return
        feats = [candidate_features(bgr, c) for c in cands]
        if self.model.ready:
            probs = self.model.predict(np.vstack(feats))
        else:
            probs = [heuristic_prob(c, f) for c, f in zip(cands, feats)]
        for c, f, p in zip(cands, feats, probs):
            c.features = f
            c.prob = float(p)

    def process(self, bgr, t=None):
        """Obradi jedan frejm (BGR). Vraca recnik sa rezultatom."""
        if t is None:
            t = time.time()
        self.frame_h, self.frame_w = bgr.shape[:2]
        cands, mask = find_candidates(bgr, self.cfg)
        self.score(bgr, cands)
        for box in list(self.exclude_boxes or []):
            x0, y0, x1, y1 = box
            for c in cands:
                if x0 <= c.x <= x1 and y0 <= c.y <= y1:
                    c.prob = 0.0          # paradajz u kantici - ne racuna se
        tr = self.tracker.update(cands, t)
        if self._t_prev is not None and t > self._t_prev:
            inst = 1.0 / (t - self._t_prev)
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
        self._t_prev = t
        res = {
            't': t,
            'w': self.frame_w,
            'h': self.frame_h,
            'detected': bool(tr is not None and tr.confirmed and (t - tr.last_seen) < 0.4),
            'tracking': tr is not None,
            'x': float(tr.x) if tr else None,
            'y': float(tr.y) if tr else None,
            'r': float(tr.r) if tr else None,
            'conf': float(tr.conf) if tr else 0.0,
            'prob': float(tr.prob) if tr else 0.0,
            'candidates': [c.as_dict() for c in cands],
            'fps': round(self.fps, 1),
        }
        with self._lock:
            self._result = res
            self.last_mask = mask
            self._last_cands = cands
        return res

    def latest(self):
        with self._lock:
            return self._result

    def last_candidates(self):
        with self._lock:
            return list(getattr(self, '_last_cands', []))

    def draw(self, img, res=None, show_mask=False):
        """Iscrtava rezultat na KOPIJU slike za video prenos."""
        if res is None:
            res = self.latest()
        if show_mask and self.last_mask is not None:
            m = cv2.resize(self.last_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            tint = img.copy()
            tint[m > 0] = (0, 255, 255)
            img = cv2.addWeighted(img, 0.55, tint, 0.45, 0)
        if not res:
            return img
        for c in res['candidates']:
            col = (0, 200, 255) if c['p'] >= self.cfg['p_candidate'] else (140, 140, 140)
            cv2.circle(img, (int(c['x']), int(c['y'])), int(c['r']), col, 1, cv2.LINE_AA)
            cv2.putText(img, '%.2f' % c['p'], (int(c['x'] - c['r']), int(c['y'] - c['r'] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        if res['x'] is not None:
            col = (0, 255, 0) if res['detected'] else (0, 255, 255)
            x, y, r = int(res['x']), int(res['y']), int(res['r'])
            cv2.circle(img, (x, y), r + 3, col, 2, cv2.LINE_AA)
            cv2.drawMarker(img, (x, y), col, cv2.MARKER_CROSS, 12, 1)
            label = 'PARADAJZ %.0f%%' % (100 * res['conf']) if res['detected'] else 'provera %.0f%%' % (100 * res['conf'])
            cv2.putText(img, label, (max(0, x - r), max(12, y + r + 16)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, col, 1, cv2.LINE_AA)
        return img


def calibrate_from_point(bgr, x, y, half=10):
    """Iz malog prozora oko (x,y) racuna robusne pragove boje.
    Vraca dict(hue_center, hue_halfwidth, s_min, v_min) ili None."""
    H, W = bgr.shape[:2]
    x, y = int(x), int(y)
    if not (0 <= x < W and 0 <= y < H):
        return None
    patch = bgr[max(0, y - half):y + half + 1, max(0, x - half):x + half + 1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    sel = hsv[hsv[:, 1] > 50]
    if len(sel) < 10:
        return None
    ang = sel[:, 0] * (2 * math.pi / 180.0)
    hc = (math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2.0) % 180
    dh = _circ_hue_offset(sel[:, 0], hc)
    hw = int(max(10, min(24, 2.5 * float(np.std(dh)) + 8)))
    s_med = float(np.median(sel[:, 1]))
    v_med = float(np.median(sel[:, 2]))
    red_off = abs(((hc + 90) % 180) - 90)   # udaljenost od ciste crvene
    if s_med < 120 or red_off > 25:
        # koza, roze sare, narandzasto-zuto... - to nije zreo cheri paradajz
        return {'error': 'Izabrana boja nije boja zrelog paradajza (S=%d, H=%d). '
                         'Tapni tačno na sredinu paradajza.' % (s_med, hc)}
    return {'hue_center': int(round(hc)) % 180, 'hue_halfwidth': hw,
            's_min': int(max(85, min(150, 0.55 * s_med))),
            'v_min': int(max(25, min(120, 0.35 * v_med)))}
