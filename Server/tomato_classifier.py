#!/usr/bin/env/python
# File name   : tomato_classifier.py
# Author      : Adnan (dodato uz Adeept PiCar-Pro)
#
# IZMENA: mali klasifikator "da li je ovo stvarno cherry paradajz" koji
# proverava SVAKI kandidat (konturu iz maske boje) pre nego sto ga
# findColor() prihvati kao metu. Ranije se cela odluka svodila na jednu
# formulu (kruznost konture), sto je bilo prelabavo - drugi okrugli/crveni
# predmeti u kadru (jabuka, paprika, tresnja, cak i neki delovi pozadine)
# su ume(li) da prodju taj test. Ovaj klasifikator je istreniran na
# desetinama pravih fotografija cherry paradajza (sa interneta) i
# desetinama fotografija "zamki" (jabuke, jagode, paprike, tresnje, ruze,
# obican (ne-cherry) paradajz, kuhinjske pozadine) - uci na OSNOVU BOJE I
# OBLIKA/TEKSTURE zajedno, ne samo na jednoj brojki kruznosti.
#
# Namerno je napisan SAMO uz pomoc cv2 i numpy (isto sto kod vec koristi)
# da se na Raspberry Pi-ju ne mora instalirati nikakva nova biblioteka
# (scikit-learn/torch/onnx...) - sam model (logisticka regresija) je
# sacuvan kao obicne brojke u tomato_clf_weights.json, a ovde se samo
# racuna feature-vektor i primenjuje w*x+b -> sigmoid.
import json
import os
import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS_PATH = os.path.join(_THIS_DIR, 'tomato_clf_weights.json')

# --- feature extraction (MORA biti identicna onoj koriscenoj pri treniranju) ---
FEAT_SIZE = 48
H_BINS, S_BINS, V_BINS = 12, 4, 4
GRAD_CELLS = 3
GRAD_ORIENTATIONS = 9
COLOR_FEAT_LEN = H_BINS * S_BINS * V_BINS
GRAD_FEAT_LEN = GRAD_CELLS * GRAD_CELLS * GRAD_ORIENTATIONS
FEAT_LEN = COLOR_FEAT_LEN + GRAD_FEAT_LEN

# podrazumevani prag verovatnoce da se kandidat prihvati kao cherry paradajz
DEFAULT_THRESHOLD = 0.55


def extract_features(bgr_crop):
    img = cv2.resize(bgr_crop, (FEAT_SIZE, FEAT_SIZE), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None,
                         [H_BINS, S_BINS, V_BINS],
                         [0, 180, 0, 256, 0, 256])
    hist = hist.flatten().astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = ang % 180.0

    cell = FEAT_SIZE // GRAD_CELLS
    grad_feats = np.zeros((GRAD_CELLS, GRAD_CELLS, GRAD_ORIENTATIONS), dtype=np.float32)
    bin_width = 180.0 / GRAD_ORIENTATIONS
    bin_idx = np.minimum((ang / bin_width).astype(np.int32), GRAD_ORIENTATIONS - 1)

    for cy in range(GRAD_CELLS):
        for cx in range(GRAD_CELLS):
            y0, y1 = cy * cell, (cy + 1) * cell
            x0, x1 = cx * cell, (cx + 1) * cell
            m = mag[y0:y1, x0:x1]
            b = bin_idx[y0:y1, x0:x1]
            for o in range(GRAD_ORIENTATIONS):
                grad_feats[cy, cx, o] = m[b == o].sum()

    grad_feats = grad_feats.flatten()
    gsum = grad_feats.sum()
    if gsum > 0:
        grad_feats = grad_feats / gsum

    return np.concatenate([hist, grad_feats]).astype(np.float32)


class _TomatoClassifier(object):
    def __init__(self):
        self.ready = False
        self.coef = None
        self.intercept = 0.0
        self.mean = None
        self.scale = None
        self._load()

    def _load(self):
        try:
            with open(_WEIGHTS_PATH, 'r') as f:
                data = json.load(f)
            self.coef = np.array(data['coef'], dtype=np.float64)
            self.intercept = float(data['intercept'])
            self.mean = np.array(data['mean'], dtype=np.float64)
            self.scale = np.array(data['scale'], dtype=np.float64)
            if self.coef.shape[0] != FEAT_LEN:
                raise ValueError('feature length mismatch: weights=%d code=%d' %
                                  (self.coef.shape[0], FEAT_LEN))
            self.ready = True
            print('[TOMATO-CLF] Ucitane tezine iz %s (feat_len=%d)' %
                  (_WEIGHTS_PATH, FEAT_LEN), flush=True)
        except Exception as e:
            # IZMENA: ako tezine nedostaju/su neispravne, klasifikator se
            # samo iskljucuje (ready=False) - findColor() se u tom slucaju
            # vraca na staru proveru kruznosti, umesto da sruzi ceo CV thread.
            self.ready = False
            print('[TOMATO-CLF] UPOZORENJE: ne mogu da ucitam %s (%s) - '
                  'klasifikator oblika je ISKLJUCEN, koristi se samo '
                  'kruznost.' % (_WEIGHTS_PATH, e), flush=True)

    def predict_proba(self, bgr_crop):
        if not self.ready or bgr_crop is None or bgr_crop.size == 0:
            return None
        try:
            feat = extract_features(bgr_crop)
            x = (feat.astype(np.float64) - self.mean) / self.scale
            z = float(np.dot(x, self.coef) + self.intercept)
            prob = 1.0 / (1.0 + np.exp(-z))
            return prob
        except Exception:
            return None


_classifier = _TomatoClassifier()


def classify_crop(bgr_crop):
    """Vraca verovatnocu (0-1) da je bgr_crop stvarno cherry paradajz,
    ili None ako klasifikator nije dostupan (fallback na kruznost)."""
    return _classifier.predict_proba(bgr_crop)


def is_available():
    return _classifier.ready
