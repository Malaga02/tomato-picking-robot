#!/usr/bin/env python3
# File name   : face_module.py
# Description : Prepoznavanje lica OSOBA SA IMENIMA (robot napise ime i mahne)
# Author      : napisano za master rad Adnana Malagica (dopuna Adeept PiCar-Pro softvera)
#
# KAKO RADI:
#   1) DETEKCIJA LICA  - YuNet (mala CNN mreza iz OpenCV Zoo, 230 KB) ako je
#                        OpenCV >= 4.8; inace Haar kaskada (Viola-Jones) koja
#                        radi na svakoj verziji OpenCV-a.
#   2) PORAVNANJE      - lice se afinom transformacijom svede na 112x112
#                        piksela tako da oci, nos i usta budu na standardnim
#                        mestima (YuNet daje 5 tacaka lica; kod Haar-a se
#                        polozaj ociju i brade procenjuje iz okvira lica).
#   3) OTISAK LICA     - SFace (OpenCV Zoo, MobileFaceNet, ~99.4% na LFW skupu)
#                        pretvara lice u vektor od 128 brojeva. Isto lice daje
#                        slicne vektore; poredjenje je kosinusna slicnost.
#   4) VISE OSOBA      - svaka osoba ima IME i svoje uzorke (otiske). Lice
#                        pripada osobi sa najvecom slicnoscu, ali samo ako je
#                        slicnost >= prag I dovoljno veca od sledece osobe
#                        (margina) - inace je "nepoznat". Odluka se potvrdjuje
#                        kroz vise uzastopnih analiza (bez "treperenja" imena).
#   5) UCENJE          - "Nauci lice" + ime: ~15 uzoraka u razlicitim polozajima
#                        glave; uzorci losijeg kvaliteta (mutni, premali) se
#                        preskacu, a uzorci koji ne lice na ostale (neko drugi u
#                        kadru) se izbacuju. Ucenje se odbija ako je to lice vec
#                        nauceno pod drugim imenom.
#
# Merenja (test_face.py, fotografije 9 razlicitih osoba, augmentacije):
#   ista osoba (DRUGE fotografije, ne one iz ucenja): slicnost 0.72 - 0.83
#   druge osobe:                                       slicnost -0.04 - 0.31
#
# Otisci i isecci lica cuvaju se SAMO lokalno na robotu (faces/), nigde se ne
# salju; osoba se brise jednim dugmetom.

import json
import os
import re
import shutil
import threading
import time
import unicodedata

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_THIS_DIR, 'models')
FACES_DIR = os.path.join(_THIS_DIR, 'faces')
SFACE_FILE = 'face_recognition_sface_2021dec.onnx'
YUNET_FILE = 'face_detection_yunet_2023mar.onnx'
HAAR_FILE = 'haarcascade_frontalface_default.xml'
HAAR_SYSTEM = [
    '/usr/share/opencv4/haarcascades/' + HAAR_FILE,
    '/usr/share/opencv/haarcascades/' + HAAR_FILE,
]
SFACE_SHA256 = '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79'

# standardni polozaji 5 tacaka lica u isecku 112x112 (ArcFace/SFace sablon):
# desno oko, levo oko, vrh nosa, desni i levi ugao usana
REF_5PTS = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)

FACE_DEFAULTS = {
    'threshold': 0.40,        # kosinusna slicnost: ispod ovoga lice je "nepoznato"
    'margin': 0.06,           # najbolja osoba mora biti bar ovoliko ispred druge
    'confirm_hits': 2,        # toliko uzastopnih analiza mora da potvrdi osobu
    'forget_after_s': 0.9,    # osoba "nestaje" ako se ne vidi ovoliko dugo
    'min_face_px': 36,        # manja lica (predaleko) se ne prepoznaju
    'min_sharpness': 18.0,    # Laplasijan isecka - mutni uzorci se ne uce
    'enroll_samples': 15,
    'enroll_timeout_s': 45.0,
    'enroll_min_gap_s': 0.3,
    'max_samples': 40,        # najvise uzoraka po osobi
}
MAX_PEOPLE = 20
NAME_MAX = 32

_CYR = dict(zip('абвгдђежзијклљмнњопрстћуфхцчџш',
                ['a', 'b', 'v', 'g', 'd', 'dj', 'e', 'z', 'z', 'i', 'j', 'k', 'l', 'lj', 'm', 'n', 'nj', 'o',
                 'p', 'r', 's', 't', 'c', 'u', 'f', 'h', 'c', 'c', 'dz', 's']))
_T = {'č': 'c', 'ć': 'c', 'š': 's', 'ž': 'z', 'đ': 'dj', 'Č': 'C', 'Ć': 'C', 'Š': 'S', 'Ž': 'Z', 'Đ': 'Dj'}
for _k, _v in _CYR.items():          # srpska cirilica -> latinica (Ђорђе -> Djordje)
    _T[_k] = _v
    _T[_k.upper()] = _v[:1].upper() + _v[1:]
_TRANSLIT = str.maketrans(_T)


def ascii_name(name):
    """OpenCV putText ne ume da iscrta č/ć/š/ž/đ - na videu se pise latinicom bez kvacica."""
    s = (name or '').translate(_TRANSLIT)
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    return s or '?'


def clean_name(name):
    name = re.sub(r'\s+', ' ', str(name or '')).strip()
    name = ''.join(ch for ch in name if ch.isprintable() and ch not in '<>"\\')
    return name[:NAME_MAX]


def assemble_model_parts(path, sha256=None):
    """SFace model (37 MB) se na robota moze preneti i u delovima
    (<ime>.part1, <ime>.part2, ...). Pri prvom pokretanju delovi se spoje u
    jedan fajl, proveri se SHA-256 otisak i delovi se obrisu."""
    if os.path.exists(path):
        return True
    import glob
    import hashlib
    parts = sorted(glob.glob(path + '.part*'), key=lambda q: int(q.rsplit('part', 1)[1] or 0))
    if not parts:
        return False
    tmp = path + '.tmp'
    h = hashlib.sha256()
    with open(tmp, 'wb') as out:
        for q in parts:
            with open(q, 'rb') as f:
                data = f.read()
            h.update(data)
            out.write(data)
    if sha256 and h.hexdigest() != sha256:
        os.remove(tmp)
        print('[FACE] Delovi modela su oštećeni/nepotpuni (SHA-256 se ne slaže) - prenesi ih ponovo.', flush=True)
        return False
    os.replace(tmp, path)
    for q in parts:
        try:
            os.remove(q)
        except OSError:
            pass
    print('[FACE] Model %s spojen iz %d dela.' % (os.path.basename(path), len(parts)), flush=True)
    return True


def _cv_version():
    try:
        return tuple(int(p) for p in cv2.__version__.split('.')[:2])
    except Exception:
        return (0, 0)


def _norm(v):
    v = np.asarray(v, np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


class FaceRecognizer(object):
    """Detekcija + prepoznavanje osoba. Bez niti - poziva ga FaceService."""

    def __init__(self, cfg=None, models_dir=MODELS_DIR, data_dir=FACES_DIR):
        self.cfg = dict(FACE_DEFAULTS)
        if cfg:
            self.cfg.update({k: v for k, v in cfg.items() if k in FACE_DEFAULTS})
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.people_file = os.path.join(data_dir, 'people.json')
        self.reason = ''
        self.detector_kind = None
        self._yunet = None
        self._haar = None
        self._net = None
        self._lock = threading.RLock()
        self.people = []               # [{'id','name','wave','created','emb': np.array (N,128)}]
        self._hits = {}                # pid -> [broj uzastopnih analiza, poslednje vreme]
        self._res = None
        # ucenje
        self.enrolling = False
        self.enroll_name = ''
        self._enroll_pid = None
        self._enroll = []
        self._enroll_crops = []
        self._enroll_yaw = []
        self._enroll_conflict = {}
        self._enroll_skipped = 0
        self._enroll_target = int(self.cfg['enroll_samples'])
        self._enroll_deadline = 0.0
        self._enroll_last = 0.0
        self.enroll_msg = ''
        self.enroll_result = None      # ('ok'|'error', poruka) - poslednji ishod ucenja
        self._init_models()
        self.load()

    # ------------------------------------------------------------------
    # Modeli
    # ------------------------------------------------------------------
    def _init_models(self):
        p = os.path.join(self.models_dir, SFACE_FILE)
        try:
            assemble_model_parts(p, SFACE_SHA256)
        except Exception as e:
            print('[FACE] Spajanje delova modela nije uspelo: %s' % e, flush=True)
        if not os.path.exists(p) and os.path.exists(p + '.part1'):
            self.reason = ('Delovi modela models/%s.part* su oštećeni ili nisu svi preneti - '
                           'prenesi ih ponovo.' % SFACE_FILE)
        elif not os.path.exists(p):
            self.reason = 'Nedostaje models/%s - prepoznavanje lica nije moguće.' % SFACE_FILE
        else:
            try:
                net = cv2.dnn.readNetFromONNX(p)
                net.setInput(cv2.dnn.blobFromImage(np.zeros((112, 112, 3), np.uint8), 1.0, (112, 112)))
                out = net.forward()
                if out.size != 128:
                    raise ValueError('neočekivan izlaz modela %s' % (out.shape,))
                self._net = net
            except Exception as e:
                self.reason = 'SFace model ne može da se učita (OpenCV %s): %s' % (cv2.__version__, e)
        yp = os.path.join(self.models_dir, YUNET_FILE)
        if hasattr(cv2, 'FaceDetectorYN') and _cv_version() >= (4, 8) and os.path.exists(yp):
            try:
                y = cv2.FaceDetectorYN.create(yp, '', (320, 240), 0.75, 0.3, 50)
                y.detect(np.zeros((240, 320, 3), np.uint8))
                self._yunet = y
                self.detector_kind = 'yunet'
            except Exception as e:
                print('[FACE] YuNet nije dostupan (%s) - koristim Haar.' % e, flush=True)
        if self._yunet is None:
            for hp in [os.path.join(self.models_dir, HAAR_FILE)] + HAAR_SYSTEM:
                if os.path.exists(hp):
                    c = cv2.CascadeClassifier(hp)
                    if not c.empty():
                        self._haar = c
                        self.detector_kind = 'haar'
                        break
        if self.detector_kind is None and not self.reason:
            self.reason = 'Nema detektora lica (ni YuNet ni Haar kaskade).'
        print('[FACE] detektor=%s prepoznavanje=%s %s' % (
            self.detector_kind, 'SFace' if self._net is not None else 'NEMA', self.reason), flush=True)

    @property
    def available(self):
        return self._net is not None and self.detector_kind is not None

    @property
    def threshold(self):
        # Haar okvir daje grublje poravnanje lica nego YuNet tacke, pa je prag
        # malo strozi (merenje: druge osobe do 0.40 sa Haar-om, do 0.31 sa YuNet-om)
        return float(self.cfg['threshold']) + (0.05 if self.detector_kind == 'haar' else 0.0)

    @property
    def enrolled(self):
        return any(len(p['emb']) >= 3 for p in self.people)

    # ------------------------------------------------------------------
    # Detekcija, poravnanje, otisak
    # ------------------------------------------------------------------
    def detect(self, bgr):
        """Lista lica: {'box': (x, y, w, h), 'lm': 5x2 ili None, 'score'}."""
        H, W = bgr.shape[:2]
        out = []
        if self._yunet is not None:
            self._yunet.setInputSize((W, H))
            _, faces = self._yunet.detect(bgr)
            if faces is not None:
                for f in faces:
                    out.append({'box': tuple(int(round(v)) for v in f[0:4]),
                                'lm': np.array(f[4:14], np.float32).reshape(5, 2),
                                'score': float(f[14])})
        elif self._haar is not None:
            s = 320.0 / W if W > 320 else 1.0
            small = cv2.resize(bgr, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s != 1.0 else bgr
            gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            rects = self._haar.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(24, 24))
            for (x, y, w, h) in rects:
                out.append({'box': (int(x / s), int(y / s), int(w / s), int(h / s)), 'lm': None, 'score': 1.0})
        out.sort(key=lambda f: f['box'][2] * f['box'][3], reverse=True)
        return out

    @staticmethod
    def align(bgr, face):
        """Isecak 112x112 poravnat na standardni sablon lica."""
        if face.get('lm') is not None:
            M, _ = cv2.estimateAffinePartial2D(face['lm'], REF_5PTS, method=cv2.LMEDS)
            if M is not None:
                return cv2.warpAffine(bgr, M, (112, 112), borderMode=cv2.BORDER_REPLICATE)
        x, y, w, h = face['box']
        src = np.float32([[x + 0.30 * w, y + 0.40 * h], [x + 0.70 * w, y + 0.40 * h], [x + 0.5 * w, y + 0.95 * h]])
        dst = np.float32([[38.29, 51.60], [73.53, 51.50], [56.0, 51.6 + (92.3 - 51.6) * 1.375]])
        M = cv2.getAffineTransform(src, dst)
        return cv2.warpAffine(bgr, M, (112, 112), borderMode=cv2.BORDER_REPLICATE)

    def embed(self, face112):
        blob = cv2.dnn.blobFromImage(face112, 1.0, (112, 112), (0, 0, 0), swapRB=True, crop=False)
        self._net.setInput(blob)
        return _norm(self._net.forward())

    @staticmethod
    def sharpness(face112):
        g = cv2.cvtColor(face112, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())

    @staticmethod
    def yaw(face):
        """Gruba procena okreta glave iz 5 tacaka (-1 levo .. +1 desno), ili None."""
        lm = face.get('lm')
        if lm is None:
            return None
        eye_mid = (lm[0] + lm[1]) / 2.0
        d = float(np.linalg.norm(lm[1] - lm[0]))
        if d < 1e-3:
            return None
        return float(np.clip((lm[2][0] - eye_mid[0]) / (0.5 * d), -1.0, 1.0))

    def identify(self, emb, exclude_pid=None):
        """(osoba ili None, najbolja slicnost, druga najbolja slicnost)."""
        with self._lock:
            people = [p for p in self.people if len(p['emb']) and p['id'] != exclude_pid]
        if not people:
            return None, None, None
        sims = [float(np.max(p['emb'] @ emb)) for p in people]
        order = np.argsort(sims)[::-1]
        best = sims[order[0]]
        second = sims[order[1]] if len(order) > 1 else -1.0
        person = people[order[0]]
        if best >= self.threshold and (best - second) >= float(self.cfg['margin']):
            return person, best, second
        return None, best, second

    # ------------------------------------------------------------------
    def analyze(self, bgr, t=None):
        """Jedna analiza frejma. Vraca i objavljuje rezultat."""
        if t is None:
            t = time.time()
        res = {'t': t, 'w': bgr.shape[1], 'h': bgr.shape[0], 'faces': [], 'people': [],
               'owner': False, 'best': None, 'enrolling': self.enrolling}
        if not self.available:
            with self._lock:
                self._res = res
            return res
        faces = self.detect(bgr)
        seen = {}
        best_any = None
        for i, f in enumerate(faces[:3]):
            x, y, w, h = f['box']
            item = {'x': x, 'y': y, 'w': w, 'h': h, 'sim': None, 'pid': None, 'name': None,
                    'small': False, 'owner': False}
            if min(w, h) < self.cfg['min_face_px']:
                item['small'] = True
                res['faces'].append(item)
                continue
            crop = self.align(bgr, f)
            emb = self.embed(crop)
            person, best, _ = self.identify(emb)
            if best is not None:
                item['sim'] = round(best, 3)
                best_any = best if best_any is None else max(best_any, best)
            if person is not None:
                item['pid'] = person['id']
                item['name'] = person['name']
                item['owner'] = True
                if person['id'] not in seen or best > seen[person['id']][1]:
                    seen[person['id']] = (person, best, item)
            res['faces'].append(item)
            if self.enrolling and i == 0:
                self._enroll_add(emb, crop, f, t)
        if self.enrolling:
            self._enroll_check(t)
        # potvrda kroz vreme: osoba mora biti vidjena u vise uzastopnih analiza
        for pid, (person, sim, item) in seen.items():
            n, _ = self._hits.get(pid, (0, 0.0))
            self._hits[pid] = (n + 1, t)
        for pid in list(self._hits):
            if pid not in seen and t - self._hits[pid][1] > float(self.cfg['forget_after_s']):
                del self._hits[pid]
        for pid, (person, sim, item) in seen.items():
            if self._hits[pid][0] >= int(self.cfg['confirm_hits']):
                item['confirmed'] = True
                res['people'].append({'pid': pid, 'name': person['name'], 'sim': round(sim, 3),
                                      'wave': bool(person.get('wave', True)),
                                      'box': (item['x'], item['y'], item['w'], item['h'])})
        res['people'].sort(key=lambda p: -p['sim'])
        res['owner'] = bool(res['people'])
        res['best'] = None if best_any is None else round(best_any, 3)
        res['enrolling'] = self.enrolling
        with self._lock:
            self._res = res
        return res

    def latest(self):
        with self._lock:
            return self._res

    def owner_box(self, res=None, pid=None):
        """Okvir (x, y, w, h) potvrdjene osobe (ili zadate osobe pid) ili None."""
        res = res or self.latest()
        if not res or not res.get('people'):
            return None
        for p in res['people']:
            if pid is None or p['pid'] == pid:
                return tuple(p['box'])
        return None

    @staticmethod
    def draw(img, res, enroll_text=None):
        if not res:
            return img
        sx = img.shape[1] / float(res.get('w') or img.shape[1])
        sy = img.shape[0] / float(res.get('h') or img.shape[0])
        for f in res['faces']:
            if f.get('name') and f.get('confirmed'):
                col, label = (60, 220, 60), '%s %d%%' % (ascii_name(f['name']), int(100 * f['sim']))
            elif f.get('name'):
                col, label = (0, 220, 255), '%s?' % ascii_name(f['name'])
            elif f['small']:
                col, label = (160, 160, 160), 'lice (daleko)'
            else:
                col = (255, 190, 0)
                label = 'nepoznat' if f['sim'] is None else 'nepoznat %d%%' % int(100 * max(0.0, f['sim']))
            x0, y0 = int(f['x'] * sx), int(f['y'] * sy)
            x1, y1 = int((f['x'] + f['w']) * sx), int((f['y'] + f['h']) * sy)
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 2, cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(th + 4, y0 - 4)
            cv2.rectangle(img, (x0, ty - th - 4), (x0 + tw + 6, ty + 3), col, -1)
            cv2.putText(img, label, (x0 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        if enroll_text:
            cv2.putText(img, enroll_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        return img

    # ------------------------------------------------------------------
    # Osobe
    # ------------------------------------------------------------------
    def people_list(self):
        with self._lock:
            return [{'id': p['id'], 'name': p['name'], 'wave': bool(p.get('wave', True)),
                     'samples': int(len(p['emb'])), 'created': p.get('created')} for p in self.people]

    def _person(self, pid):
        for p in self.people:
            if p['id'] == pid:
                return p
        return None

    def rename(self, pid, name):
        name = clean_name(name)
        if not name:
            return False, 'Ime ne sme biti prazno.'
        with self._lock:
            p = self._person(pid)
            if p is None:
                return False, 'Osoba ne postoji.'
            if any(q['name'].lower() == name.lower() and q['id'] != pid for q in self.people):
                return False, 'Osoba sa imenom "%s" već postoji.' % name
            old = p['name']
            p['name'] = name
            self._save()
        return True, 'Ime promenjeno: %s → %s.' % (old, name)

    def set_wave(self, pid, on):
        with self._lock:
            p = self._person(pid)
            if p is None:
                return False, 'Osoba ne postoji.'
            p['wave'] = bool(on)
            self._save()
        return True, ('Robot će mahati osobi %s.' if on else 'Robot neće mahati osobi %s.') % p['name']

    def delete(self, pid):
        with self._lock:
            p = self._person(pid)
            if p is None:
                return False, 'Osoba ne postoji.'
            self.people = [q for q in self.people if q['id'] != pid]
            self._hits.pop(pid, None)
            self._save()
        d = os.path.join(self.data_dir, 'people', pid)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
        return True, 'Osoba %s obrisana sa robota.' % p['name']

    def forget(self):
        """Brise SVE osobe (i stari format sa jednim vlasnikom)."""
        self.enrolling = False
        with self._lock:
            self.people = []
            self._hits = {}
        for sub in ('people', 'owner'):
            d = os.path.join(self.data_dir, sub)
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
        for name in ('people.json', 'owner_embeddings.json', 'owner_model.json',
                     'owner_embeddings.json.migrated'):
            p = os.path.join(self.data_dir, name)
            if os.path.exists(p):
                os.remove(p)
        self.enroll_msg = 'Sve osobe su obrisane sa robota.'
        return self.enroll_msg

    # ------------------------------------------------------------------
    # Ucenje lica
    # ------------------------------------------------------------------
    def start_enroll(self, name=None, pid=None, n=None, timeout=None):
        if not self.available:
            return False, 'Prepoznavanje lica nije dostupno: %s' % self.reason
        with self._lock:
            if pid is not None:
                p = self._person(pid)
                if p is None:
                    return False, 'Osoba ne postoji.'
                name = p['name']
            else:
                name = clean_name(name)
                if not name:
                    return False, 'Upiši ime osobe pre učenja lica.'
                ex = [q for q in self.people if q['name'].lower() == name.lower()]
                if ex:
                    pid = ex[0]['id']          # isto ime -> dopuni uzorke te osobe
                    name = ex[0]['name']
                elif len(self.people) >= MAX_PEOPLE:
                    return False, 'Najviše %d osoba - obriši neku pre dodavanja nove.' % MAX_PEOPLE
        self.enroll_name = name
        self._enroll_pid = pid
        self._enroll = []
        self._enroll_crops = []
        self._enroll_yaw = []
        self._enroll_conflict = {}
        self._enroll_skipped = 0
        self._enroll_target = int(n or self.cfg['enroll_samples'])
        self._enroll_deadline = time.time() + float(timeout or self.cfg['enroll_timeout_s'])
        self._enroll_last = 0.0
        self.enroll_result = None
        self.enrolling = True
        self.enroll_msg = ('Učim lice: %s. Gledaj u kameru sa 0,5–1 m i polako okreći glavu '
                           'malo levo-desno i gore-dole.' % name)
        return True, self.enroll_msg

    def cancel_enroll(self):
        if self.enrolling:
            self.enrolling = False
            self.enroll_msg = 'Učenje lica otkazano.'
            self.enroll_result = ('error', self.enroll_msg)
        return self.enroll_msg

    @property
    def enroll_progress(self):
        return len(self._enroll), self._enroll_target

    @property
    def enroll_coverage(self):
        """Koliko uzoraka je snimljeno sa glavom levo / pravo / desno."""
        y = [v for v in self._enroll_yaw if v is not None]
        return {'levo': sum(v < -0.2 for v in y), 'pravo': sum(-0.2 <= v <= 0.2 for v in y),
                'desno': sum(v > 0.2 for v in y)}

    def _enroll_add(self, emb, crop, face, t):
        if t - self._enroll_last < self.cfg['enroll_min_gap_s']:
            return
        if self.sharpness(crop) < self.cfg['min_sharpness']:
            self._enroll_skipped += 1
            return
        self._enroll_last = t
        other, sim, _ = self.identify(emb, exclude_pid=self._enroll_pid)
        if other is not None:
            self._enroll_conflict[other['name']] = self._enroll_conflict.get(other['name'], 0) + 1
        self._enroll.append(emb)
        self._enroll_crops.append(crop)
        self._enroll_yaw.append(self.yaw(face))

    def _enroll_check(self, t):
        n = len(self._enroll)
        if n >= self._enroll_target:
            self._finish_enroll()
        elif t > self._enroll_deadline:
            if n >= 6:
                self._finish_enroll()
            else:
                self.enrolling = False
                self.enroll_msg = ('Učenje prekinuto: lice je snimljeno samo %d puta%s. Pokušaj ponovo '
                                   '(više svetla, lice bliže kameri).' % (
                                       n, ' (%d mutnih uzoraka preskočeno)' % self._enroll_skipped
                                       if self._enroll_skipped else ''))
                self.enroll_result = ('error', self.enroll_msg)

    def _finish_enroll(self):
        self.enrolling = False
        E = np.array(self._enroll, np.float32)
        crops = list(self._enroll_crops)
        # isto lice vec postoji pod drugim imenom?
        if self._enroll_conflict:
            nm, cnt = max(self._enroll_conflict.items(), key=lambda kv: kv[1])
            if cnt >= 0.4 * len(E):
                self.enroll_msg = ('Ovo lice je već naučeno kao "%s". Ako je to ista osoba, '
                                   'izaberi je na listi i dopuni uzorke.' % nm)
                self.enroll_result = ('error', self.enroll_msg)
                return
        center = _norm(np.median(E, axis=0))
        keep = (E @ center) >= 0.45
        if keep.sum() < 6:
            self.enroll_msg = ('Uzorci lica su previše različiti (verovatno više osoba u kadru). '
                               'Pokušaj ponovo - samo jedna osoba ispred kamere.')
            self.enroll_result = ('error', self.enroll_msg)
            return
        E = E[keep]
        crops = [c for c, k in zip(crops, keep) if k]
        with self._lock:
            if self._enroll_pid is not None and self._person(self._enroll_pid) is not None:
                p = self._person(self._enroll_pid)
                allE = np.vstack([p['emb'], E]) if len(p['emb']) else E
                p['emb'] = allE[-int(self.cfg['max_samples']):]
                msg = 'Dopunjeno lice: %s (%d uzoraka).' % (p['name'], len(p['emb']))
            else:
                pid = 'p%x' % int(time.time() * 1000)
                p = {'id': pid, 'name': self.enroll_name, 'wave': True,
                     'created': time.strftime('%Y-%m-%d %H:%M'), 'emb': E}
                self.people.append(p)
                msg = 'Lice naučeno: %s (%d uzoraka). Robot će napisati ime kad te prepozna.' % (
                    p['name'], len(E))
            self._hits = {}
            try:
                self._save()
                self._save_crops(p['id'], crops)
            except Exception as e:
                msg += ' (Snimanje na disk nije uspelo: %s)' % e
        self.enroll_msg = msg
        self.enroll_result = ('ok', msg)

    # ------------------------------------------------------------------
    # Cuvanje
    # ------------------------------------------------------------------
    def _save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        data = {'version': 3, 'model': SFACE_FILE,
                'people': [{'id': p['id'], 'name': p['name'], 'wave': bool(p.get('wave', True)),
                            'created': p.get('created'),
                            'embeddings': np.round(p['emb'], 6).tolist()} for p in self.people]}
        tmp = self.people_file + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.people_file)

    def _save_crops(self, pid, crops):
        d = os.path.join(self.data_dir, 'people', pid)
        os.makedirs(d, exist_ok=True)
        start = len([f for f in os.listdir(d) if f.endswith('.jpg')])
        for i, c in enumerate(crops):
            cv2.imwrite(os.path.join(d, '%03d.jpg' % (start + i + 1)), c)

    def load(self):
        try:
            with open(self.people_file, encoding='utf-8') as f:
                d = json.load(f)
            if d.get('version') != 3 or d.get('model') != SFACE_FILE:
                raise ValueError('drugi format')
            people = []
            for p in d.get('people', []):
                E = np.array(p['embeddings'], np.float32)
                if E.ndim != 2 or E.shape[1] != 128:
                    continue
                people.append({'id': str(p['id']), 'name': clean_name(p['name']) or '?',
                               'wave': bool(p.get('wave', True)), 'created': p.get('created'),
                               'emb': np.array([_norm(e) for e in E], np.float32)})
            self.people = people
            print('[FACE] Učitano osoba: %d (%s).' % (len(people), ', '.join(p['name'] for p in people)), flush=True)
            return
        except FileNotFoundError:
            pass
        except Exception as e:
            print('[FACE] people.json nije učitan: %s' % e, flush=True)
        self._migrate_v2()

    def _migrate_v2(self):
        """Prethodna verzija je pamtila samo jedno lice (vlasnik) - prebaci ga
        kao osobu "Vlasnik" (ime se menja na stranici)."""
        old = os.path.join(self.data_dir, 'owner_embeddings.json')
        try:
            with open(old) as f:
                d = json.load(f)
            if d.get('model') != SFACE_FILE:
                return
            E = np.array(d['embeddings'], np.float32)
            if E.ndim != 2 or E.shape[1] != 128:
                return
            self.people = [{'id': 'p0', 'name': 'Vlasnik', 'wave': True, 'created': d.get('created'),
                            'emb': np.array([_norm(e) for e in E], np.float32)}]
            self._save()
            os.replace(old, old + '.migrated')
            print('[FACE] Lice iz prethodne verzije prebačeno kao osoba "Vlasnik".', flush=True)
        except FileNotFoundError:
            pass
        except Exception as e:
            print('[FACE] Stari model lica nije prebačen: %s' % e, flush=True)


class FaceService(threading.Thread):
    """Pozadinska nit: kad je potrebno (prepoznavanje ukljuceno ili ucenje
    lica), nekoliko puta u sekundi uzme NAJNOVIJI frejm sa kamere i analizira
    ga. Radi odvojeno od video prenosa, pa sporija analiza ne usporava sliku.

    frame_source() -> (bgr, t) ili (None, 0)
    on_owner(res)  -> poziva se kad je bar jedna osoba potvrdjena
    """

    def __init__(self, recognizer, frame_source, on_owner=None, interval=0.35):
        super(FaceService, self).__init__()
        self.daemon = True
        self.rec = recognizer
        self.frame_source = frame_source
        self.on_owner = on_owner
        self.interval = interval
        self.enabled = False       # prepoznavanje (imena / pozdrav) ukljuceno
        self.suspended = False     # npr. dok traje berba (CPU i servoi su zauzeti)
        self.last_ms = 0.0
        self.last_error = ''
        self._last_t = 0.0
        self._wake = threading.Event()

    @property
    def active(self):
        return self.rec.available and (self.rec.enrolling or (self.enabled and not self.suspended))

    def poke(self):
        self._wake.set()

    def step(self):
        """Jedna iteracija (izdvojeno radi testiranja). Vraca rezultat ili None."""
        if not self.active:
            return None
        frame, t = self.frame_source()
        if frame is None or t <= self._last_t:
            return None
        self._last_t = t
        t0 = time.time()
        res = self.rec.analyze(frame, t)
        self.last_ms = 1000.0 * (time.time() - t0)
        if res['people'] and self.on_owner is not None and not self.rec.enrolling:
            try:
                self.on_owner(res)
            except Exception as e:
                self.last_error = str(e)
        return res

    def run(self):
        while True:
            if not self.active:
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            t0 = time.time()
            try:
                self.step()
            except Exception as e:
                self.last_error = str(e)
                print('[FACE] greška: %s' % e, flush=True)
                time.sleep(1.0)
            time.sleep(max(0.05, self.interval - (time.time() - t0)))
