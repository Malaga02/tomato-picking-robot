#!/usr/bin/env python3
# File name   : bucket_detector.py
# Description : Prepoznavanje zelene kantice u koju robot ubacuje paradajz
# Author      : napisano za master rad Adnana Malagica (dopuna Adeept PiCar-Pro softvera)
#
# KAKO RADI:
#   1) MASKA BOJE   - pikseli cija je nijansa (H) blizu zelene boje kantice i
#                     dovoljno su zasiceni (S) i svetli (V). Boja se po potrebi
#                     kalibrise dodirom na kanticu na kontrolnoj stranici.
#   2) ZAKLONJENI   - kamera je na glavi ruke, pa se u donjem delu kadra UVEK
#      DELOVI         vide prsti hvataljke (i paradajz koji robot nosi). Kad se
#                     robot priblizi, oni zaklone donji deo kantice i mrlja
#                     dobije "zarez" - stara verzija je tada kanticu gubila
#                     (glavni razlog zasto robot nije ubacivao paradajz).
#                     Sada se zaklonjena oblast (maska hvataljke: kalibrisana
#                     i/ili automatski procenjena kao tamna oblast vezana za
#                     donju ivicu kadra + paradajz u hvataljci) tretira kao
#                     "nepoznato": u proveri oblika se racuna kao deo
#                     kantice (unutar njenog obrisa), a ne racuna se u
#                     proveri boje ni dodira sa ivicom kadra.
#   3) OBLIK        - kantica je velika, "puna" i kompaktna mrlja ujednacene
#                     boje i glatke povrsine (lisce/trava su nepravilni i hrapavi).
#   4) PRACENJE     - potvrda tek posle nekoliko frejmova; kad je kantica vec
#                     potvrdjena, prihvata se i veca i "secena" ivicama kadra
#                     (robot je tada blizu) - ali samo na mestu gde je bila.
#
# Modul ne zavisi od hardvera - testira se nad slikama (test_bucket.py,
# test_bucket_approach.py).

import math
import threading
import time

import cv2
import numpy as np

BUCKET_DEFAULTS = {
    'proc_width': 320,
    # OpenCV H je 0..179; zelena kantica sa fotografija: H ~ 44-56
    'hue_center': 50,
    'hue_halfwidth': 16,
    's_min': 80,
    'v_min': 40,
    'min_area_frac': 0.003,
    'max_area_frac': 0.60,
    'min_solidity': 0.80,
    'min_extent': 0.55,
    'max_hue_std': 9.0,
    'min_aspect': 0.35,
    'max_texture': 0.10,
    'max_borders': 2,
    # blaze granice za VEC PRACENU kanticu (robot je blizu, kantica je velika)
    'trk_max_area_frac': 0.98,
    'trk_min_solidity': 0.62,
    'trk_min_extent': 0.42,
    'trk_max_borders': 4,
    # automatska procena prstiju hvataljke (tamno, vezano za donju ivicu)
    'auto_claw': True,
    'claw_v_max': 70,
    'claw_region_y': 0.45,       # prsti se traze samo u donjem delu kadra
    'confirm_hits': 3,
    'max_unseen_s': 1.0,
    'ema': 0.5,
}


def _hue_dist(h, center):
    return np.abs(((h.astype(np.int16) - int(center) + 90) % 180) - 90)


def auto_claw_mask(hsv, cfg=BUCKET_DEFAULTS):
    """Procena gde su prsti hvataljke: tamni pikseli u donjem delu kadra koji
    su povezani sa donjom ivicom. Vraca uint8 masku (0/1)."""
    h, w = hsv.shape[:2]
    dark = (hsv[:, :, 2] < cfg['claw_v_max']).astype(np.uint8)
    dark[:int(h * cfg['claw_region_y'])] = 0
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    out = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if y + bh >= h - 2 and area >= 0.004 * h * w:
            out[lab == i] = 1
    if out.any():
        out = cv2.dilate(out, np.ones((5, 5), np.uint8))
    return out


def calibrate_claw_mask(frames, v_max=90, std_max=9.0):
    """KALIBRACIJA MASKE HVATALJKE: frejmovi snimljeni dok se ruka (sa
    kamerom) okrece levo-desno. Scena se u kadru pomera, a prsti hvataljke
    stoje na istom mestu -> pikseli sa malom promenom kroz vreme koji su
    tamni i vezani za ivicu kadra su hvataljka. Vraca (maska 0/255, poruka)."""
    if len(frames) < 6:
        return None, 'Premalo frejmova sa kamere (otvori video na stranici).'
    g = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames])
    std = g.std(axis=0)
    mean = g.mean(axis=0)
    h, w = mean.shape
    if float(np.median(std)) < 4.0:
        return None, ('Scena se skoro ne menja dok se ruka okreće (jednobojna podloga?). '
                      'Postavi robota iznad šare/tepiha pa ponovi.')
    m = ((std < std_max) & (mean < v_max)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        touches = y + bh >= h - 2 or x <= 1 or x + bw >= w - 1
        if touches and area >= 0.003 * h * w:
            out[lab == i] = 255
    frac = float((out > 0).mean())
    if frac < 0.005:
        return None, 'Prsti hvataljke nisu pronađeni u kadru (da li se vide na kameri?).'
    if frac > 0.5:
        return None, 'Previše nepromenljivih tamnih piksela - ponovi kalibraciju na svetlijoj, šarenoj podlozi.'
    out = cv2.dilate(out, np.ones((5, 5), np.uint8))
    return out, 'Maska hvataljke snimljena (%.0f%% kadra).' % (100 * frac)


class BucketDetector(object):
    def __init__(self, cfg=None):
        self.cfg = dict(BUCKET_DEFAULTS)
        if cfg:
            self.cfg.update({k: v for k, v in cfg.items() if k in BUCKET_DEFAULTS})
        self._lock = threading.Lock()
        self._res = None
        self._track = None
        self.last_mask = None
        self.last_ignore = None

    # ------------------------------------------------------------------
    def mask(self, small, hsv=None):
        """Maska boje kantice (bez premoscavanja) sa popunjenim rupama."""
        if hsv is None:
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        return self._fill(self._clean(self._color(hsv))), hsv

    def _color(self, hsv):
        c = self.cfg
        d = _hue_dist(hsv[:, :, 0], c['hue_center'])
        return ((d <= c['hue_halfwidth']) & (hsv[:, :, 1] >= c['s_min']) &
                (hsv[:, :, 2] >= c['v_min'])).astype(np.uint8) * 255

    @staticmethod
    def _clean(m):
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k7, iterations=2)

    @staticmethod
    def _fill(m):
        cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        filled = np.zeros_like(m)
        cv2.drawContours(filled, cnts, -1, 255, -1)
        return filled

    @staticmethod
    def _bridge(m, ign):
        """Zelena se 'nastavlja' preko zaklonjene oblasti samo u istom redu:
        zelena | zaklonjeno ... zaklonjeno | zelena (prsti/paradajz ISPRED
        kantice, sa vidljivom kanticom levo i desno). Zarez odozdo se ne
        "dopisuje" - on se uracunava u proveri oblika (vidi candidates)."""
        g = m > 0
        h, w = g.shape
        from_l = np.zeros_like(g)
        from_r = np.zeros_like(g)
        prev = g[:, 0].copy()
        for x in range(1, w):
            prev = ign[:, x] & (prev | g[:, x - 1])
            from_l[:, x] = prev
        prev = g[:, w - 1].copy()
        for x in range(w - 2, -1, -1):
            prev = ign[:, x] & (prev | g[:, x + 1])
            from_r[:, x] = prev
        out = m.copy()
        out[from_l & from_r] = 255
        return out

    def candidates(self, bgr, ignore=None, relaxed=False):
        """Kandidati za kanticu. ignore: maska (0/>0) zaklonjenih piksela u
        rezoluciji frejma. relaxed=True vraca i kandidate koji prolaze samo
        blaze granice (polje 'strict' kaze da li prolaze i stroge)."""
        c = self.cfg
        H, W = bgr.shape[:2]
        pw = int(c['proc_width'])
        scale = W / float(pw) if W > pw else 1.0
        small = cv2.resize(bgr, (int(round(W / scale)), int(round(H / scale))),
                           interpolation=cv2.INTER_AREA) if scale != 1.0 else bgr
        sh, sw = small.shape[:2]
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        raw = self._clean(self._color(hsv))
        ign = np.zeros((sh, sw), bool)
        if ignore is not None:
            ig = cv2.resize(ignore.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
            ign |= ig > 0
        if c.get('auto_claw', True):
            ign |= auto_claw_mask(hsv, c) > 0
        if ign.any():
            # malo prosiri (ivice prstiju/paradajza su pomesane boje - bez
            # ovoga izmedju zelene i zaklonjenog ostane uzan "procep")
            ign = cv2.dilate(ign.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        ign &= raw == 0                       # vidljiva zelena nikad nije "zaklonjena"
        m = self._fill(self._bridge(raw, ign) if ign.any() else raw)
        self.last_ignore = ign
        cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        g = lap = None
        out = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            frac = area / float(sw * sh)
            if frac < c['min_area_frac'] or frac > c['trk_max_area_frac']:
                continue
            hull_pts = cv2.convexHull(cnt)
            hull = cv2.contourArea(hull_pts)
            x, y, w, h = cv2.boundingRect(cnt)
            reg = np.zeros((sh, sw), np.uint8)
            cv2.drawContours(reg, [cnt], -1, 255, -1)
            inside = reg > 0
            # zaklonjeni pikseli unutar konveksnog omotaca / pravougaonika se
            # racunaju kao kantica (prsti hvataljke ispred nje, "zarez" odozdo)
            occ_hull = occ_rect = 0
            if ign.any():
                hm = np.zeros((sh, sw), np.uint8)
                cv2.drawContours(hm, [hull_pts], -1, 255, -1)
                occ_hull = int(np.count_nonzero((hm > 0) & ign & ~inside))
                occ_rect = int(np.count_nonzero(ign[y:y + h, x:x + w] & ~inside[y:y + h, x:x + w]))
            solidity = min(1.0, (area + occ_hull) / hull) if hull > 0 else 0.0
            extent = min(1.0, (area + occ_rect) / float(w * h)) if w * h else 0.0
            aspect = min(w, h) / float(max(w, h)) if max(w, h) else 0.0
            if aspect < c['min_aspect'] or solidity < c['trk_min_solidity'] or extent < c['trk_min_extent']:
                continue
            vis = inside & ~ign                 # stvarno vidljivi pikseli kantice
            if vis.sum() < 10:
                continue
            occluded = 1.0 - (vis.sum() / float(inside.sum() + occ_hull))
            # dodir sa ivicom kadra - samo vidljivim (ne zaklonjenim) pikselima
            borders = (int(vis[:, :2].any()) + int(vis[:2, :].any()) +
                       int(vis[:, -2:].any()) + int(vis[-2:, :].any()))
            strict_pix = vis & (hsv[:, :, 1] >= c['s_min'])
            if strict_pix.sum() < 10:
                continue
            hd = ((hsv[:, :, 0][strict_pix].astype(np.int16) - int(c['hue_center']) + 90) % 180) - 90
            hue_std = float(np.std(hd))
            if hue_std > c['max_hue_std']:
                continue
            core = (cv2.erode(reg, np.ones((5, 5), np.uint8)) > 0) & ~ign
            if core.sum() >= 10:
                if g is None:
                    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (3, 3), 0), cv2.CV_32F))
                texture = float(np.mean(lap[core])) / (float(np.mean(g[core])) + 10.0)
            else:
                texture = 0.0
            if texture > c['max_texture']:
                continue
            strict = (frac <= c['max_area_frac'] and solidity >= c['min_solidity'] and
                      extent >= c['min_extent'] and borders <= c['max_borders'])
            if not strict and (not relaxed or borders > c['trk_max_borders']):
                continue
            score = frac * solidity * extent / (1.0 + 0.1 * hue_std)
            out.append({
                'x': (x + w / 2.0) * scale, 'y': (y + h / 2.0) * scale,
                'w': w * scale, 'h': h * scale,
                'top': y * scale, 'bottom': (y + h) * scale,
                'area_frac': frac, 'solidity': solidity, 'extent': extent,
                'hue_std': hue_std, 'texture': texture, 'score': score,
                'borders': borders, 'occluded': occluded, 'strict': strict,
            })
        out.sort(key=lambda d: (d['strict'], d['score']), reverse=True)
        self.last_mask = m
        return out

    # ------------------------------------------------------------------
    def process(self, bgr, t=None, ignore=None):
        if t is None:
            t = time.time()
        H, W = bgr.shape[:2]
        c = self.cfg
        tr = self._track
        confirmed = tr is not None and tr['hits'] >= c['confirm_hits']
        cands = self.candidates(bgr, ignore=ignore, relaxed=confirmed)
        best = None
        if tr is not None and cands:
            def near(d):
                # isti objekat: centar blizu ili se okviri znatno preklapaju
                if math.hypot(d['x'] - tr['x'], d['y'] - tr['y']) < 0.6 * max(tr['w'], tr['h'], 40):
                    return True
                ix = (min(d['x'] + d['w'] / 2, tr['x'] + tr['w'] / 2) -
                      max(d['x'] - d['w'] / 2, tr['x'] - tr['w'] / 2))
                iy = (min(d['y'] + d['h'] / 2, tr['y'] + tr['h'] / 2) -
                      max(d['y'] - d['h'] / 2, tr['y'] - tr['h'] / 2))
                return ix > 0 and iy > 0 and ix * iy > 0.3 * min(d['w'] * d['h'], tr['w'] * tr['h'])
            close = [d for d in cands if near(d)]
            if close:
                best = close[0]
        if best is None:
            strict = [d for d in cands if d['strict']]
            best = strict[0] if strict else None
            if best is not None and tr is not None:
                tr = None          # druga, nova kantica - pocni pracenje iz pocetka
        if best is not None:
            if tr is None:
                tr = dict(best)
                tr['hits'] = 1
                tr['max_w'] = best['w']
            else:
                a = c['ema']
                for k in ('x', 'y', 'w', 'h', 'top', 'bottom'):
                    tr[k] = a * best[k] + (1 - a) * tr[k]
                tr['hits'] += 1
                tr['max_w'] = max(tr.get('max_w', 0), tr['w'])
            tr['last'] = t
            tr['occluded'] = best['occluded']
            tr['borders'] = best['borders']
            self._track = tr
        elif tr is not None and t - tr.get('last', t) > c['max_unseen_s']:
            self._track = tr = None
        res = {
            't': t, 'w_img': W, 'h_img': H,
            'detected': bool(tr is not None and tr['hits'] >= c['confirm_hits'] and t - tr['last'] < 0.5),
            'x': tr['x'] if tr else None, 'y': tr['y'] if tr else None,
            'bw': tr['w'] if tr else None, 'bh': tr['h'] if tr else None,
            'top': tr['top'] if tr else None, 'bottom': tr['bottom'] if tr else None,
            'max_w': tr.get('max_w') if tr else None,
            'occluded': tr.get('occluded', 0.0) if tr else 0.0,
            'borders': tr.get('borders', 0) if tr else 0,
            'last_seen': tr['last'] if tr else None,
            'n': len(cands),
        }
        with self._lock:
            self._res = res
        return res

    def latest(self):
        with self._lock:
            return self._res

    def reset(self):
        self._track = None

    def draw(self, img, res=None):
        if res is None:
            res = self.latest()
        if not res or res['x'] is None:
            return img
        col = (60, 220, 60) if res['detected'] else (0, 200, 255)
        x0 = int(res['x'] - res['bw'] / 2)
        y0 = int(res['y'] - res['bh'] / 2)
        x1 = int(res['x'] + res['bw'] / 2)
        y1 = int(res['y'] + res['bh'] / 2)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 2, cv2.LINE_AA)
        label = 'KANTICA' if res['detected'] else 'kantica?'
        if res.get('occluded', 0) > 0.05:
            label += ' (zaklonjeno %d%%)' % int(100 * res['occluded'])
        cv2.putText(img, label, (max(0, x0), max(12, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        return img

    # ------------------------------------------------------------------
    def set_color(self, **kw):
        for k in ('hue_center', 'hue_halfwidth', 's_min', 'v_min'):
            if kw.get(k) is not None:
                self.cfg[k] = int(kw[k])
        self.cfg['hue_center'] %= 180
        self.cfg['hue_halfwidth'] = int(max(8, min(30, self.cfg['hue_halfwidth'])))
        self.cfg['s_min'] = int(max(50, min(200, self.cfg['s_min'])))
        self.cfg['v_min'] = int(max(20, min(160, self.cfg['v_min'])))
        self.reset()


def calibrate_bucket_color(bgr, x, y, half=14):
    """Boja kantice iz prozora oko (x, y). Vraca dict ili {'error': ...}."""
    H, W = bgr.shape[:2]
    x, y = int(x), int(y)
    if not (0 <= x < W and 0 <= y < H):
        return {'error': 'Tačka je van slike.'}
    patch = bgr[max(0, y - half):y + half + 1, max(0, x - half):x + half + 1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    sel = hsv[hsv[:, 1] > 50]
    if len(sel) < 20:
        return {'error': 'Na tom mestu nema dovoljno zasićene boje - tapni na sredinu kantice.'}
    ang = sel[:, 0] * (2 * math.pi / 180.0)
    hc = (math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2.0) % 180
    red_off = abs(((hc + 90) % 180) - 90)
    if red_off < 20:
        return {'error': 'Ta boja je crvena/narandžasta (paradajz?) - tapni na KANTICU.'}
    dh = _hue_dist(sel[:, 0], hc)
    s_med = float(np.median(sel[:, 1]))
    v_med = float(np.median(sel[:, 2]))
    return {'hue_center': int(round(hc)) % 180,
            'hue_halfwidth': int(max(10, min(24, 2.5 * float(np.std(dh)) + 8))),
            's_min': int(max(50, min(160, 0.5 * s_med))),
            'v_min': int(max(25, min(120, 0.4 * v_med)))}
