#!/usr/bin/env python3
# File name   : TomatoPicker.py
# Description : Autonomno i rucno branje cheri paradajza, odlaganje u kanticu
#               ili na pozu odlaganja, prepoznavanje osoba po imenu i pozdrav
# Author      : napisano za master rad Adnana Malagica (dopuna Adeept PiCar-Pro softvera)
#
# HARDVER:
#   Kamera je na glavi ruke (zajedno sa hvataljkom): kanal 1 okrece ruku
#   levo-desno (pan), kanal 2 je ruka gore/dole, kanal 3 zglob (nagib kamere
#   i hvataljke), kanal 4 hvataljka, kanal 0 skretanje prednjih tockova.
#
# PODELA POSLA:
#   CV nit (camera_opencv.py, rezim 'tomato') na svakom frejmu radi detekciju
#   paradajza (tomato_detector.py) i kantice (bucket_detector.py) i samo
#   OBJAVLJUJE rezultat. Servo motore i motore tockova pomera iskljucivo ova
#   nit (ili, van berbe, JEDNA "idle" nit za mahanje/test/kalibraciju) - nikad
#   dve niti istovremeno. Lica analizira posebna nit (face_module.FaceService)
#   samo kad je prepoznavanje ukljuceno, a berba ne radi.
#
# REZIMI:
#   AUTOMATSKI - "Pokreni berbu": robot sam trazi, bere i odlaze paradajz u
#                krug ("Uberi jedan" = samo jedan ciklus).
#   RUCNI      - robota vozis i ruku pomeras sam, a robot izvrsava pojedinacne
#                korake: "Uhvati" (zatvori-podigni-proveri), "Ubaci u kanticu"
#                (nadji kanticu, pridji, ubaci), "Odlozi" (poza odlaganja).
#
# CIKLUS BERBE (stanja):
#   TRAZENJE -> CENTRIRANJE -> PRILAZAK -> SPUSTANJE -> NAVODJENJE -> HVATANJE
#   -> (kantica ukljucena i kalibrisana) TRAZENJE_KANTICE -> PRILAZAK_KANTICI
#      -> UBACIVANJE
#   -> (kantica iskljucena / nije kalibrisana / nije nadjena) ODLAGANJE na
#      snimljenu POZU ODLAGANJA (okret ruke, ruka, zglob).
#
# SIGURNOST:
#   tockovi staju cim meta nije vidljiva ili slika kasni; STOP prekida svaku
#   fazu i svaki pokret odmah; svaka faza ima vremensko ogranicenje;
#   ultrazvucni senzor je dodatna kocnica; posle STOP-a robot se glatko vraca
#   u pocetni polozaj (samo ako je bio u pokretu). Podesavanja: tomato_config.py.

import collections
import math
import os
import threading
import time

import cv2
import numpy as np

import Move as move
from camera_opencv import Camera, CVThread
import assistant
import bucket_detector
import tomato_config
import tomato_detector

try:
    import Ultra as ultra
    ULTRA_OK = True
except Exception:
    ULTRA_OK = False

try:
    import face_module
    FACE_IMPORT_ERROR = ''
except Exception as _e:          # npr. star OpenCV bez cv2.dnn
    face_module = None
    FACE_IMPORT_ERROR = str(_e)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(_THIS_DIR, 'calib')


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class TomatoPicker(threading.Thread):
    STATE_IDLE = 'IDLE'
    STATE_SEARCH = 'TRAZENJE'
    STATE_CENTER = 'CENTRIRANJE'
    STATE_APPROACH = 'PRILAZAK'
    STATE_LOWER = 'SPUSTANJE'
    STATE_FINE = 'NAVODJENJE'
    STATE_GRAB = 'HVATANJE'
    STATE_FIND_BUCKET = 'TRAZENJE_KANTICE'
    STATE_BUCKET_APPROACH = 'PRILAZAK_KANTICI'
    STATE_DROP = 'UBACIVANJE'
    STATE_DEPOSIT = 'ODLAGANJE'
    CARRY_STATES = ('TRAZENJE_KANTICE', 'PRILAZAK_KANTICI', 'UBACIVANJE', 'ODLAGANJE')

    # epizode: 'full' = berba u krug, 'one' = jedan ciklus, rucni koraci:
    # 'grab' = uhvati ovde, 'deliver' = ubaci u kanticu, 'place' = odlozi
    EPISODES = ('full', 'one', 'grab', 'deliver', 'place')

    def __init__(self, scGear, *args, **kwargs):
        super(TomatoPicker, self).__init__(*args, **kwargs)
        self.daemon = True
        self.scGear = scGear
        self.cfg = tomato_config.load()
        self.P = self.cfg['picker']
        self.F = self.cfg['face']
        self.U = self.cfg['ui']
        self.running = False
        self.episode = 'full'
        self.state = self.STATE_IDLE
        self.lastError = ''
        self.lastInfo = ''
        self.pan = 0.0
        self.steer = 0.0
        self._searchDir = 1
        self._t_cmd = 0.0
        self._last_seen = 0.0
        self._approach_start = 0.0
        self._last_res_t = 0.0
        self._last_bucket_t = 0.0
        self._need_pose = False
        self._ang = {}            # poslednji zadati ugao po kanalu (glatki pokreti)
        self._grab_tries = 0
        self._fine_backups = 0
        self.holding = False      # paradajz je u hvataljci (ne otvaraj je pri mahanju)
        self._fine_start = 0.0
        self._phase_start = 0.0
        self._bucket_search_start = 0.0
        self._bucket_max_w = 0.0
        self._sweeps = 0
        self._log_t = 0.0
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopRequested = threading.Event()
        # statistika sesije
        self.pickedCount = 0      # ukupno odlozenih paradajza
        self.bucketCount = 0      # od toga u kanticu
        self.attempts = 0         # zapocetih prilazaka paradajzu
        self.failedGrabs = 0
        self.waveCount = 0
        self.stats = collections.Counter()
        self._run_started = None
        self._run_total = 0.0
        self._cycle_start = None
        self.cycleTimes = collections.deque(maxlen=20)
        # dnevnik dogadjaja za stranicu (id raste, stranica trazi samo nove)
        self._events = collections.deque(maxlen=250)
        self._event_id = 0
        self._event_lock = threading.Lock()
        self.bootId = '%x' % int(time.time() * 1000)   # stranica prepoznaje restart servera
        # pokreti VAN berbe (mahanje, test hvataljke, povratak posle STOP-a,
        # kalibracija maske hvataljke) idu kroz JEDNU "idle" nit
        self._idleAbort = threading.Event()
        self._idleThread = None
        self._idleKind = ''
        self._busy = False          # nit berbe je usred faze
        self._starting = False      # START je u toku (ne pocinji nove pokrete)
        self._last_wave = 0.0
        self._greeted = {}          # pid -> vreme poslednjeg pozdrava
        self.greeting = None        # poslednje prepoznavanje {'pid','name','sim','t'}
        self._ultra_fail = 0
        self.ultraOk = None
        self._bright = (0.0, None)
        # maske hvataljke (kalibrisane) - za prepoznavanje zaklonjene kantice
        self._claw_masks = {}
        self._load_claw_masks()
        self.face = None
        self.faceService = None
        self.faceError = FACE_IMPORT_ERROR
        self._init_face()
        self._publish_grab_zone()
        self._update_bucket_ignore()
        self._event('info', 'Sistem spreman.')

    # ==================================================================
    # Pomocne
    # ==================================================================
    def _event(self, level, text):
        with self._event_lock:
            self._event_id += 1
            self._events.append({'id': self._event_id, 't': time.time(), 'level': level, 'text': text})

    def _info(self, text):
        if text and text != self.lastInfo:
            self.lastInfo = text
            self._event('info', text)

    def _error(self, text):
        if text and text != self.lastError:
            self._event('warn', text)
        self.lastError = text

    def _det(self):
        return Camera.tomato_det

    def _bucket(self):
        return Camera.bucket_det

    def _servo(self, ch, angle):
        try:
            self.scGear.moveAngle(ch, angle)
        except Exception as e:
            self.lastError = 'servo %d: %s' % (ch, e)

    def _current_angle(self, ch):
        g = self.scGear
        return int(round((g.nowPos[ch] - g.initPos[ch]) * g.sc_direction[ch]))

    def _cur(self, ch):
        if ch not in self._ang:
            try:
                self._ang[ch] = float(self._current_angle(ch))
            except Exception:
                self._ang[ch] = 0.0
        return self._ang[ch]

    def _move(self, ch, target, dps=None, interruptible=True, evt=None):
        """GLATKO pomeranje servoa do ciljnog ugla zadatom brzinom (stepeni u
        sekundi). Prekida se odmah ako se postavi 'evt' (podrazumevano STOP
        berbe). Vraca False ako je prekinuto."""
        P = self.P
        evt = self._stopRequested if evt is None else evt
        if dps is None:
            dps = {0: 90, 1: P.get('pan_speed_dps', 45), 2: P.get('arm_speed_dps', 35),
                   3: P.get('wrist_speed_dps', 45), 4: P.get('claw_speed_dps', 40)}.get(ch, 40)
        target = float(target)
        a0 = self._cur(ch)
        dist = abs(target - a0)
        dt = 0.03
        n = max(1, int(math.ceil(dist / max(dps * dt, 0.5))))
        for i in range(1, n + 1):
            a = a0 + (target - a0) * i / n
            self._servo(ch, int(round(a)))
            self._ang[ch] = a
            if ch == 1:
                self.pan = a
            if i < n:
                if interruptible:
                    if evt.wait(dt):
                        return False
                else:
                    time.sleep(dt)
        self._ang[ch] = target
        if ch == 1:
            self.pan = target
            self._t_cmd = time.time()
        return (not interruptible) or not evt.is_set()

    def _set_pan(self, angle, smooth=True):
        lim = self.P['pan_limit']
        a = _clamp(angle, -lim, lim)
        if smooth:
            self._move(1, a)
        else:
            self._servo(1, int(round(a)))
            self._ang[1] = a
            self.pan = a
        self._t_cmd = time.time()

    def _set_steer(self, angle):
        m = self.P['steer_max']
        a = int(round(_clamp(angle, -m, m)))
        if a != int(round(self.steer)):
            self._servo(0, a)
            self._ang[0] = a
        self.steer = a

    def _stop_motors(self):
        try:
            move.motorStop()
        except Exception as e:
            self.lastError = 'motori: %s' % e

    def _sleep(self, s, evt=None):
        """Prekidivo spavanje - vraca False ako je u medjuvremenu stigao STOP."""
        evt = self._stopRequested if evt is None else evt
        return not evt.wait(timeout=max(0.0, s))

    def _fresh(self, after_t, timeout=0.8, source=None):
        """Ceka rezultat detekcije sa NOVOG frejma snimljenog POSLE after_t (da
        se ne reaguje na sliku od pre poslednjeg pokreta, niti dva puta na isti
        frejm). None ako ga nema."""
        end = time.time() + timeout
        det = source or self._det()
        bucket = det is self._bucket()
        while not self._stopRequested.is_set():
            r = det.latest()
            last = self._last_bucket_t if bucket else self._last_res_t
            if r is not None and r['t'] >= after_t and r['t'] > last:
                if bucket:
                    self._last_bucket_t = r['t']
                else:
                    self._last_res_t = r['t']
                return r
            if time.time() > end:
                return None
            time.sleep(0.015)
        return None

    def _ultra(self):
        if not ULTRA_OK:
            self.ultraOk = False
            return None
        try:
            d = ultra.checkdist()
        except Exception:
            d = -1
        if d is not None and d > 0:
            self._ultra_fail = 0
            self.ultraOk = True
            return d
        self._ultra_fail += 1
        if self._ultra_fail >= 8:
            self.ultraOk = False
        return None

    def _look_pose(self, interruptible=True, evt=None):
        """Glatko u pozu gledanja (ruka, zglob, otvorena hvataljka, tockovi pravo)."""
        P = self.P
        self._set_steer(0)
        ok = self._move(2, P['look_arm'], interruptible=interruptible, evt=evt)
        ok = ok and self._move(3, P['look_wrist'], interruptible=interruptible, evt=evt)
        ok = ok and self._move(4, P['claw_open'], interruptible=interruptible, evt=evt)
        if ok:
            self.holding = False
        ok = ok and self._move(1, self.pan, interruptible=interruptible, evt=evt)
        return ok

    def _holding_tomato(self):
        """Da li hvataljka (verovatno) drzi paradajz: zastavica iz berbe I
        hvataljka je stvarno blize zatvorenom polozaju (korisnik je mozda
        rucno otvorio hvataljku)."""
        if not self.holding:
            return False
        P = self.P
        try:
            a = float(self._current_angle(4))
        except Exception:
            return True
        return abs(a - P['claw_closed']) < abs(a - P['claw_open'])

    def _grab_pose(self):
        P = self.P
        ok = self._move(4, P['claw_open'])
        ok = ok and self._move(3, P['grab_wrist'])
        ok = ok and self._move(2, P['grab_arm'], P.get('arm_down_dps', 25))
        return ok

    def _carry_pose(self):
        """Poza nosenja (paradajz u hvataljci, kamera trazi kanticu)."""
        P = self.P
        arm = P['look_arm'] if P.get('carry_arm') is None else P['carry_arm']
        wrist = P['look_wrist'] if P.get('carry_wrist') is None else P['carry_wrist']
        ok = self._move(2, arm, P.get('arm_lift_dps', 10))
        return ok and self._move(3, wrist, P.get('wrist_speed_dps', 20))

    def _log(self, r=None, force=False):
        """Kratak zapis u journal (journalctl -u Adeept_Robot) - najvise 1x/s."""
        now = time.time()
        if not force and now - self._log_t < 1.0:
            return
        self._log_t = now
        if self.state in self.CARRY_STATES:
            b = self._bucket().latest() or {}
            print('[TOMATO-PICK] %s kantica det=%s x=%s w=%s max_w=%s zaklonjeno=%.2f pan=%.0f steer=%d info=%s err=%s' % (
                self.state, int(bool(b.get('detected'))), None if b.get('x') is None else int(b['x']),
                None if b.get('bw') is None else int(b['bw']), None if b.get('max_w') is None else int(b['max_w']),
                b.get('occluded', 0.0) or 0.0, self.pan, self.steer, self.lastInfo, self.lastError), flush=True)
            return
        if r is None:
            r = self._det().latest() or {}
        print('[TOMATO-PICK] %s det=%s trk=%s x=%s y=%s r=%s conf=%.2f pan=%.0f steer=%d info=%s err=%s' % (
            self.state, int(bool(r.get('detected'))), int(bool(r.get('tracking'))),
            None if r.get('x') is None else int(r['x']), None if r.get('y') is None else int(r['y']),
            None if r.get('r') is None else int(r['r']), r.get('conf', 0.0) or 0.0,
            self.pan, self.steer, self.lastInfo, self.lastError), flush=True)

    def _grab_point(self, res):
        P = self.P
        w, h = res['w'], res['h']
        return P['grab_x_frac'] * w, P['grab_y_frac'] * h, P['grab_r_frac'] * w

    def _set_state(self, st, info=''):
        # posle STOP-a faza koja se upravo prekida ne sme da "prepise" IDLE
        if self._stopRequested.is_set() and st != self.STATE_IDLE:
            return
        changed = st != self.state
        if changed:
            self.state = st
            Camera.picker_state = st
            self._phase_start = time.time()
            self._update_bucket_ignore()
        if info:
            self._info(info)
        if changed:
            self._log(force=True)

    def _reload_cfg(self):
        self.cfg = tomato_config.load()
        self.P = self.cfg['picker']
        self.F = self.cfg['face']
        self.U = self.cfg['ui']

    def _set_cfg(self, sec, key, value):
        cfg = tomato_config.load()
        cfg[sec][key] = value
        tomato_config.save(cfg)
        self._reload_cfg()

    # ==================================================================
    # Maska hvataljke (zaklonjeni deo kadra pri prepoznavanju kantice)
    # ==================================================================
    def _load_claw_masks(self):
        self._claw_masks = {}
        for k in ('open', 'closed'):
            p = os.path.join(CALIB_DIR, 'claw_%s.png' % k)
            if os.path.exists(p):
                m = cv2.imread(p, 0)
                if m is not None:
                    self._claw_masks[k] = m > 0

    def _update_bucket_ignore(self):
        """Objavi CV niti koji deo kadra je zaklonjen prstima hvataljke (i
        paradajzom koji robot nosi) - kantica iza njih se tada ne "gubi"."""
        carrying = self.state in self.CARRY_STATES or self.episode in ('deliver', 'place') and self.running
        base = self._claw_masks.get('closed' if carrying else 'open')
        H, W = (base.shape if base is not None else
                (int(CVThread.videoH or 240), int(CVThread.videoW or 320)))
        m = np.zeros((H, W), np.uint8) if base is None else base.astype(np.uint8)
        if carrying:
            P = self.P
            cv2.circle(m, (int(P['grab_x_frac'] * W), int(P['grab_y_frac'] * H)),
                       max(4, int(P['grab_r_frac'] * W * 1.3)), 1, -1)
        Camera.bucket_ignore = m if m.any() else None

    def calibrateClawMask(self):
        """KALIBRACIJA MASKE HVATALJKE: robot (stoji u mestu) polako okrece
        ruku levo-desno sa otvorenom, pa sa zatvorenom hvataljkom; iz frejmova
        nalazi piksele koji se ne menjaju (prsti hvataljke)."""
        if self.running:
            return 'Zaustavi berbu pre kalibracije maske hvataljke.'
        if Camera.last_raw is None or time.time() - Camera.last_raw_t > 1.5:
            return 'Kamera ne šalje slike - otvori video na stranici.'
        if not self._run_idle('clawmask', self._calib_claw_mask):
            return 'Robot je upravo zauzet drugim pokretom.'
        return 'Kalibracija maske hvataljke: robot okreće ruku levo-desno (~8 s), ne pomeraj ga.'

    def _calib_claw_mask(self):
        evt, P = self._idleAbort, self.P
        start_pan = self._cur(1)
        lim = P['pan_limit']
        msgs, masks = [], {}
        aborted = True
        try:
            for state, ang in (('open', P['claw_open']), ('closed', P['claw_closed'])):
                if not self._move(4, ang, evt=evt) or evt.wait(0.4):
                    return
                frames, t_last = [], 0.0
                for tgt in (start_pan - 15, start_pan + 15, start_pan):
                    tgt = _clamp(tgt, -lim, lim)
                    a0 = self._cur(1)
                    n = max(1, int(abs(tgt - a0) / 1.5))
                    for i in range(1, n + 1):
                        a = a0 + (tgt - a0) * i / float(n)
                        self._servo(1, int(round(a)))
                        self._ang[1] = a
                        if evt.wait(0.07):
                            return
                        f, t = Camera.last_raw, Camera.last_raw_t
                        if f is not None and t > t_last:
                            frames.append(f.copy())
                            t_last = t
                mask, msg = bucket_detector.calibrate_claw_mask(frames)
                label = 'otvorena' if state == 'open' else 'zatvorena'
                msgs.append('%s: %s' % (label, msg))
                if mask is not None:
                    masks[state] = mask
            aborted = False
            # fajlovi se upisuju tek kad je ceo postupak zavrsen (bez polovicnog stanja)
            os.makedirs(CALIB_DIR, exist_ok=True)
            for state, mask in masks.items():
                cv2.imwrite(os.path.join(CALIB_DIR, 'claw_%s.png' % state), mask)
        finally:
            if not aborted:
                self._move(4, P['claw_open'], evt=evt)   # (posle STOP-a se nista ne pomera)
            self._load_claw_masks()
            self._update_bucket_ignore()
            if aborted:
                self._event('warn', 'Kalibracija maske hvataljke prekinuta - stara maska je zadržana.')
            else:
                ok = len(self._claw_masks) == 2
                self._event('ok' if ok else 'warn', 'Maska hvataljke - ' + '; '.join(msgs))

    def resetClawMask(self):
        for k in ('open', 'closed'):
            p = os.path.join(CALIB_DIR, 'claw_%s.png' % k)
            if os.path.exists(p):
                os.remove(p)
        self._load_claw_masks()
        self._update_bucket_ignore()
        return 'Maska hvataljke obrisana (koristi se samo automatska procena).'

    # ==================================================================
    # Javni interfejs (WebServer.py)
    # ==================================================================
    def preview(self):
        """Ukljuci detekciju (prikaz oznaka na videu) i kad berba ne radi."""
        if Camera.modeSelect in ('none', 'tomato'):
            Camera.modeSelect = 'tomato'

    # ---------------- pokreti van berbe (jedna nit, prekidivi) ----------------
    def _run_idle(self, kind, target, *args):
        """Pokrene pokret van berbe u posebnoj niti. Vraca False ako vec
        radi neki drugi pokret ili traje berba."""
        with self._lock:
            if self.running or self._starting or (self._idleThread is not None and self._idleThread.is_alive()):
                return False
            self._idleAbort.clear()
            self._idleKind = kind
            self._ang = {}       # servoi su mozda pomereni rucno - kreni od stvarnih uglova
            th = threading.Thread(target=self._idle_wrap, args=(target,) + args, daemon=True)
            self._idleThread = th
            th.start()
        return True

    def _idle_wrap(self, target, *args):
        try:
            target(*args)
        except Exception as e:
            self._error('Pokret (%s): %s' % (self._idleKind, e))

    def _abort_idle(self):
        self._idleAbort.set()
        th = self._idleThread
        if th is not None and th.is_alive() and th is not threading.current_thread():
            th.join(timeout=3.0)

    def _abort_wave(self):        # (staro ime - kompatibilnost)
        self._abort_idle()

    @property
    def waving(self):
        th = self._idleThread
        return self._idleKind == 'wave' and th is not None and th.is_alive()

    @property
    def idleBusy(self):
        th = self._idleThread
        return self._idleKind if th is not None and th.is_alive() else ''

    def startPicking(self, episode='full'):
        """episode: 'full' (u krug), 'one' (jedan paradajz), 'grab', 'deliver', 'place'."""
        if episode not in self.EPISODES:
            return 'Nepoznat zadatak.'
        if episode in ('full', 'one') and self.U.get('control_mode') == 'manual':
            return 'Robot je u RUČNOM režimu - prebaci na automatski za samostalnu berbu.'
        if self.faceService is not None:
            self.faceService.suspended = True   # pre svega: nema novih pozdrava
        with self._lock:
            self._starting = True
        try:
            self._abort_idle()
            return self._start_locked(episode)
        finally:
            self._starting = False

    def _start_locked(self, episode):
        with self._lock:
            if self.running:
                return 'Robot već radi.'
            self._reload_cfg()
            self._stopRequested.clear()
            self.episode = episode
            self.running = True
            self.lastError = ''
            self._ang = {}
            self._grab_tries = 0
            self._fine_backups = 0
            self._sweeps = 0
            self._run_started = time.time()
            self._cycle_start = time.time()
            if self.faceService is not None:
                self.faceService.suspended = True
            Camera.modeSelect = 'tomato'
            self._det().tracker.reset()
            self._bucket().reset()
            if episode in ('full', 'one'):
                self.pan = 0.0
                self._need_pose = True
                self._set_state(self.STATE_SEARCH, 'Berba pokrenuta - tražim paradajz.' if episode == 'full'
                                else 'Berem jedan paradajz.')
            else:
                self.pan = self._cur(1)
                self._need_pose = False
                if episode == 'grab':
                    self._grab_tries = max(0, int(self.P['max_grab_attempts']) - 1)   # rucno: bez ponavljanja voznje
                    self._set_state(self.STATE_GRAB, 'Ručno: hvatam paradajz ovde.')
                elif episode == 'deliver':
                    if self._deliver_target() == 'bucket':
                        self._begin_bucket_search('Ručno: tražim kanticu.', carry=False)
                    else:
                        self._set_state(self.STATE_DEPOSIT, 'Ručno: kantica nije kalibrisana - odlažem na pozu odlaganja.')
                else:
                    self._set_state(self.STATE_DEPOSIT, 'Ručno: odlažem paradajz na pozu odlaganja.')
            self._wake.set()
        return {'full': 'Berba pokrenuta.', 'one': 'Berem jedan paradajz.', 'grab': 'Hvatam.',
                'deliver': 'Nosim u kanticu.', 'place': 'Odlažem.'}[episode]

    def stopPicking(self):
        self._stopRequested.set()
        self._abort_idle()
        with self._lock:
            was = self.running
            self.running = False
            if self._run_started is not None:
                self._run_total += time.time() - self._run_started
                self._run_started = None
            self._set_state(self.STATE_IDLE, 'Zaustavljeno.' if was else '')
            self._wake.clear()
        self._stop_motors()
        Camera.modeSelect = 'tomato'   # oznake na videu ostaju ukljucene
        if was and self.episode in ('full', 'one'):
            # glatki povratak u pocetni polozaj (prekida ga novi START);
            # STOP dok robot miruje ili u rucnom koraku NE pomera ruku
            self.pan = 0.0
            self._run_idle('home', self._home_after_stop)
        elif self.faceService is not None:
            self.faceService.suspended = False
        self._update_bucket_ignore()
        return 'Zaustavljeno.'

    def _home_after_stop(self):
        evt = self._idleAbort
        t0 = time.time()
        while self._busy and time.time() - t0 < 3.0 and not evt.is_set():
            time.sleep(0.05)
        self._stop_motors()
        self._ang = {}
        self._look_pose(evt=evt)
        if self.faceService is not None and not self.running:
            self.faceService.suspended = False
            self.faceService.poke()

    def _finish_episode(self, text, level='ok'):
        """Rucni korak / jedan ciklus je zavrsen - robot staje BEZ povratka."""
        with self._lock:
            self.running = False
            if self._run_started is not None:
                self._run_total += time.time() - self._run_started
                self._run_started = None
            self._wake.clear()
            changed = self.state != self.STATE_IDLE
            self.state = self.STATE_IDLE
            Camera.picker_state = self.STATE_IDLE
            if changed:
                self._phase_start = time.time()
        self._stop_motors()
        self._update_bucket_ignore()
        self.lastInfo = text
        self._event(level, text)
        if self.faceService is not None:
            self.faceService.suspended = False
            self.faceService.poke()

    def resetStats(self):
        self.pickedCount = self.bucketCount = self.attempts = self.failedGrabs = self.waveCount = 0
        self.stats.clear()
        self._run_total = 0.0
        if self._run_started is not None:
            self._run_started = time.time()
        self.cycleTimes.clear()
        self._event('info', 'Statistika obrisana.')
        return 'Statistika obrisana.'

    # ---------------- rezimi / prekidaci ----------------
    def setControlMode(self, mode):
        if mode not in ('auto', 'manual'):
            return 'Nepoznat režim.'
        if mode == 'manual' and self.running and self.episode in ('full', 'one'):
            self.stopPicking()
        self._set_cfg('ui', 'control_mode', mode)
        msg = 'Ručni režim: ti voziš i pomeraš ruku, robot izvršava korake na dugme.' if mode == 'manual' \
            else 'Automatski režim: robot sam traži, bere i odlaže paradajz.'
        self._event('info', msg)
        return msg

    def setBucketMode(self, on):
        self._set_cfg('picker', 'deposit_mode', 'bucket' if on else 'pose')
        if on and not self.P.get('bucket_calibrated'):
            return ('Kantica uključena, ali NIJE kalibrisana - dok je ne kalibrišeš, robot odlaže na pozu '
                    'odlaganja.')
        return 'Ubacivanje u kanticu uključeno.' if on else 'Kantica isključena - robot odlaže paradajz na pozu odlaganja.'

    def setAssistant(self, on):
        self._set_cfg('ui', 'assistant', bool(on))
        return 'AI asistent uključen.' if on else 'AI asistent isključen - radiš ručno, bez preporuka.'

    def setVoice(self, on):
        self._set_cfg('ui', 'voice', bool(on))
        return 'Glasovne poruke uključene.' if on else 'Glasovne poruke isključene.'

    def _deliver_target(self):
        """'bucket' samo ako je kantica ukljucena I kalibrisana, inace 'pose'."""
        if self.P.get('deposit_mode', 'bucket') != 'bucket':
            return 'pose'
        if not self.P.get('bucket_calibrated'):
            if not self.stats.get('warnedBucketCalib'):
                self.stats['warnedBucketCalib'] = 1
                self._error('Kantica nije kalibrisana - paradajz odlažem na pozu odlaganja.')
            return 'pose'
        return 'bucket'

    # ---------------- boja paradajza ----------------
    def setTomatoColor(self, h, s, v):
        """Kompatibilnost sa starim porukama: rucno zadata boja."""
        self._det().set_color(hue_center=h, s_min=max(60, 0.55 * s), v_min=max(25, 0.35 * v))
        self._save_detector_color()

    def calibrateColorAt(self, x, y):
        frame = Camera.last_raw
        if frame is None:
            return 'Nema slike sa kamere.'
        p = tomato_detector.calibrate_from_point(frame, x, y)
        if p is None:
            return 'Na tom mestu nema dovoljno zasićene boje - dodirni sam paradajz.'
        if 'error' in p:
            return p['error'] + ' Boja NIJE promenjena.'
        self._det().set_color(**p)
        self._det().tracker.reset()
        self._save_detector_color()
        return 'Boja paradajza podešena: H=%d (±%d), S≥%d, V≥%d' % (
            p['hue_center'], p['hue_halfwidth'], p['s_min'], p['v_min'])

    def resetColor(self):
        d = tomato_detector.DETECTOR_DEFAULTS
        self._det().set_color(d['hue_center'], d['hue_halfwidth'], d['s_min'], d['v_min'])
        cfg = tomato_config.load()
        cfg['detector'] = {}
        tomato_config.save(cfg)
        return 'Boja paradajza vraćena na podrazumevanu (crvena).'

    def _save_detector_color(self):
        c = self._det().cfg
        cfg = tomato_config.load()
        cfg['detector'].update({k: c[k] for k in ('hue_center', 'hue_halfwidth', 's_min', 'v_min')})
        tomato_config.save(cfg)

    # ---------------- boja kantice ----------------
    def calibrateBucketColorAt(self, x, y):
        frame = Camera.last_raw
        if frame is None:
            return 'Nema slike sa kamere.'
        p = bucket_detector.calibrate_bucket_color(frame, x, y)
        if 'error' in p:
            return p['error'] + ' Boja kantice NIJE promenjena.'
        self._bucket().set_color(**p)
        cfg = tomato_config.load()
        cfg['bucket'] = {k: self._bucket().cfg[k] for k in ('hue_center', 'hue_halfwidth', 's_min', 'v_min')}
        tomato_config.save(cfg)
        return 'Boja kantice podešena: H=%d (±%d), S≥%d, V≥%d' % (
            p['hue_center'], p['hue_halfwidth'], p['s_min'], p['v_min'])

    def resetBucketColor(self):
        d = bucket_detector.BUCKET_DEFAULTS
        self._bucket().set_color(hue_center=d['hue_center'], hue_halfwidth=d['hue_halfwidth'],
                                 s_min=d['s_min'], v_min=d['v_min'])
        cfg = tomato_config.load()
        cfg['bucket'] = {}
        tomato_config.save(cfg)
        return 'Boja kantice vraćena na podrazumevanu (zelena).'

    # ---------------- kalibracija poza ----------------
    def saveGrabTarget(self):
        """KALIBRACIJA HVATANJA (jedno dugme): ruka je spustena u polozaj u kome
        hvataljka treba da se zatvori, hvataljka je otvorena, a paradajz stoji
        TACNO izmedju prstiju. Pamti se poza ruke/zgloba i gde se paradajz vidi
        u kadru i koliko je velik (cilj finog navodjenja)."""
        r = self._det().latest()
        if r and time.time() - r['t'] > 1.0:
            return 'Slika sa kamere nije sveža - otvori video na ovoj stranici i pokušaj ponovo.'
        if not r or r.get('x') is None or not (r['detected'] or r['tracking']):
            return ('Paradajz se ne vidi na kameri. Spusti ruku tako da kamera vidi '
                    'paradajz između otvorenih prstiju, pa pokušaj ponovo.')
        cfg = tomato_config.load()
        P = cfg['picker']
        P['grab_x_frac'] = round(r['x'] / r['w'], 4)
        P['grab_y_frac'] = round(r['y'] / r['h'], 4)
        P['grab_r_frac'] = round(r['r'] / r['w'], 4)
        P['grab_arm'] = self._current_angle(2)
        P['grab_wrist'] = self._current_angle(3)
        P['grab_calibrated'] = True
        tomato_config.save(cfg)
        self._reload_cfg()
        self._publish_grab_zone()
        return ('Hvatanje kalibrisano: paradajz na x=%.0f y=%.0f r=%.0f px, ruka %d°, zglob %d°.'
                % (r['x'], r['y'], r['r'], P['grab_arm'], P['grab_wrist']))

    def _publish_grab_zone(self):
        P = self.P
        Camera.grab_zone = (P['grab_x_frac'], P['grab_y_frac'], P['grab_r_frac'],
                            bool(P.get('grab_calibrated')))

    def _save_pose(self, keys, label, extra=None):
        cfg = tomato_config.load()
        vals = {}
        for key, ch in keys:
            cfg[key[0]][key[1]] = self._current_angle(ch)
            vals[ch] = cfg[key[0]][key[1]]
        for (sec, k), v in (extra or {}).items():
            cfg[sec][k] = v
        tomato_config.save(cfg)
        self._reload_cfg()
        parts = []
        for ch, nm in ((1, 'okret'), (2, 'ruka'), (3, 'zglob')):
            if ch in vals:
                parts.append('%s %d°' % (nm, vals[ch]))
        return '%s zapamćena (%s).' % (label, ', '.join(parts))

    def saveGrabPose(self):
        return self._save_pose([(('picker', 'grab_arm'), 2), (('picker', 'grab_wrist'), 3)], 'Poza hvatanja')

    def saveLookPose(self):
        return self._save_pose([(('picker', 'look_arm'), 2), (('picker', 'look_wrist'), 3)], 'Poza gledanja')

    def saveDropPose(self):
        """Poza ubacivanja: hvataljka iznad otvora kantice."""
        return self._save_pose([(('picker', 'drop_arm'), 2), (('picker', 'drop_wrist'), 3)], 'Poza ubacivanja u kanticu')

    def savePlacePose(self):
        """POZA ODLAGANJA: okret ruke (kanal 1), ruka i zglob tacno onako kako
        treba da stoje kad robot ostavi paradajz (npr. u korpu na robotu ili
        na pod pored njega)."""
        return self._save_pose([(('picker', 'place_pan'), 1), (('picker', 'place_arm'), 2),
                                (('picker', 'place_wrist'), 3)], 'Poza odlaganja',
                               extra={('picker', 'place_calibrated'): True})

    def saveWatchPose(self):
        return self._save_pose([(('face', 'watch_arm'), 2), (('face', 'watch_wrist'), 3)], 'Poza za lica (kamera gleda lica)')

    def goToPose(self, which):
        """Rucna provera snimljene poze: robot polako ode u nju."""
        P = self.P
        poses = {
            'look': (None, P['look_arm'], P['look_wrist']),
            'grab': (None, P['grab_arm'], P['grab_wrist']),
            'drop': (None, P.get('drop_arm'), P.get('drop_wrist')),
            'place': (P.get('place_pan'), P.get('place_arm'), P.get('place_wrist')),
            'watch': (None, self.F.get('watch_arm'), self.F.get('watch_wrist')),
        }
        if which not in poses:
            return 'Nepoznata poza.'
        pan, arm, wrist = poses[which]
        if arm is None:
            return 'Ta poza još nije snimljena.'
        if self.running:
            return 'Zaustavi robota pre provere poze.'

        def run():
            evt = self._idleAbort
            if pan is not None and not self._move(1, pan, evt=evt):
                return
            if not self._move(3, wrist, evt=evt):
                return
            self._move(2, arm, P.get('arm_speed_dps', 25), evt=evt)
        if not self._run_idle('pose', run):
            return 'Robot je upravo zauzet drugim pokretom.'
        return 'Idem u snimljenu pozu.'

    def saveBucketTarget(self):
        """KALIBRACIJA KANTICE (jedno dugme): robot stoji ispred kantice tacno
        na rastojanju na kom treba da ubaci paradajz, ruka je u pozi u kojoj
        kamera vidi kanticu. Pamti se koliko je kantica tada siroka u kadru,
        gde je, poza nosenja i (ako radi) ultrazvucno rastojanje."""
        b = self._bucket().latest()
        if b and time.time() - b['t'] > 1.0:
            return 'Slika sa kamere nije sveža - otvori video na ovoj stranici i pokušaj ponovo.'
        if not b or not b.get('detected'):
            return ('Kantica se ne vidi (nema zelenog okvira "KANTICA"). Postavi robota ispred '
                    'kantice tako da je cela u kadru; ako treba, podesi boju kantice dodirom.')
        cfg = tomato_config.load()
        P = cfg['picker']
        P['bucket_near_w_frac'] = round(_clamp(b['bw'] / float(b['w_img']), 0.08, 0.95), 4)
        P['bucket_x_frac'] = round(b['x'] / float(b['w_img']), 4)
        P['carry_arm'] = self._current_angle(2)
        P['carry_wrist'] = self._current_angle(3)
        d = self._ultra()
        if d is not None and 3 < d < 60:
            P['bucket_ultra_cm'] = round(_clamp(d - 2.0, 4.0, 40.0), 1)
        P['bucket_calibrated'] = True
        tomato_config.save(cfg)
        self._reload_cfg()
        return ('Kantica kalibrisana: širina %.0f%% kadra, sredina na %.0f%%, poza nošenja ruka %d° / zglob %d°%s.'
                % (100 * P['bucket_near_w_frac'], 100 * P['bucket_x_frac'], P['carry_arm'], P['carry_wrist'],
                   '' if d is None else ', ultrazvuk %.0f cm' % d))

    def setShowMask(self, on):
        Camera.tomato_show_mask = bool(on)

    def setLogging(self, on):
        Camera.datalog.configure(enabled=bool(on))
        cfg = tomato_config.load()
        cfg['picker']['log_enabled'] = bool(on)
        tomato_config.save(cfg)

    # ---------------- podesavanja (tab "Podesavanja") ----------------
    def getSettings(self):
        return tomato_config.settings_schema(tomato_config.load())

    def applySettings(self, changes):
        try:
            tomato_config.apply_settings(changes)
        except ValueError as e:
            return False, str(e)
        self._reload_cfg()
        self._apply_face_cfg()
        return True, 'Podešavanja sačuvana (%d).' % len(changes)

    def resetSettings(self):
        tomato_config.reset_settings()
        self._reload_cfg()
        self._apply_face_cfg()
        return 'Podešavanja vraćena na podrazumevana (kalibracije su sačuvane).'

    # ---------------- hvataljka ----------------
    def testClaw(self):
        """Otvori -> postepeno zatvori -> otvori (provera hvataljke)."""
        if self.running:
            return 'Zaustavi berbu pre testa hvataljke.'

        def run():
            evt = self._idleAbort
            if not self._move(4, self.P['claw_open'], evt=evt) or evt.wait(0.4):
                return
            if not self._move(4, self.P['claw_closed'], self.P.get('claw_close_dps', 15), evt=evt) or evt.wait(1.0):
                return
            self._move(4, self.P['claw_open'], evt=evt)
        if not self._run_idle('claw', run):
            return 'Robot je upravo zauzet drugim pokretom - pokušaj za par sekundi.'
        return 'Test hvataljke: otvoreno %d°, zatvoreno %d°.' % (self.P['claw_open'], self.P['claw_closed'])

    def saveClaw(self, which):
        cfg = tomato_config.load()
        ang = self._current_angle(4)
        key = 'claw_closed' if which == 'closed' else 'claw_open'
        cfg['picker'][key] = ang
        tomato_config.save(cfg)
        self._reload_cfg()
        return 'Zapamćeno: hvataljka %s = %d°.' % ('zatvorena' if which == 'closed' else 'otvorena', ang)

    # ==================================================================
    # Osobe: prepoznavanje po imenu i pozdrav mahanjem
    # ==================================================================
    def _init_face(self):
        if face_module is None:
            self.faceError = 'face_module se ne učitava: %s' % FACE_IMPORT_ERROR
            return
        try:
            self.face = face_module.FaceRecognizer(self.F)
            self.faceService = face_module.FaceService(
                self.face, self._frame_source, on_owner=self._on_people,
                interval=float(self.F.get('interval_s', 0.35)))
            self._sync_face_service()
            self.faceService.start()
            Camera.face_rec = self.face
            self.faceError = self.face.reason
        except Exception as e:
            self.face = self.faceService = None
            self.faceError = 'Prepoznavanje lica nije pokrenuto: %s' % e
            print('[FACE] %s' % self.faceError, flush=True)

    def _sync_face_service(self):
        if self.faceService is not None:
            self.faceService.enabled = bool(self.F.get('recognize') or self.F.get('wave_enabled'))
            self.faceService.poke()

    def _apply_face_cfg(self):
        if self.face is not None:
            self.face.cfg.update({k: v for k, v in self.F.items() if k in self.face.cfg})
        if self.faceService is not None:
            self.faceService.interval = float(self.F.get('interval_s', 0.35))
        self._sync_face_service()

    @staticmethod
    def _frame_source():
        return Camera.last_raw, Camera.last_raw_t

    def _face_ok(self):
        return self.faceService is not None and self.face is not None and self.face.available

    def setRecognize(self, on):
        if not self._face_ok():
            return 'Prepoznavanje lica nije dostupno: %s' % (self.faceError or 'nepoznat razlog')
        self._set_cfg('face', 'recognize', bool(on))
        self._sync_face_service()
        if on and not self.running and self.F.get('watch_arm') is not None:
            self._run_idle('watch', self._go_watch_pose)
        return 'Prepoznavanje osoba uključeno - robot piše ime kad prepozna lice.' if on \
            else 'Prepoznavanje osoba isključeno.'

    def setWave(self, on):
        """Ukljuci/iskljuci pozdrav mahanjem (pamti se i posle restarta)."""
        if not self._face_ok():
            return 'Prepoznavanje lica nije dostupno: %s' % (self.faceError or 'nepoznat razlog')
        self._set_cfg('face', 'wave_enabled', bool(on))
        self._sync_face_service()
        if not on:
            if self.waving:
                self._abort_idle()
            return 'Mahanje isključeno.'
        msg = 'Mahanje uključeno - robot maše osobama kojima je to dozvoljeno.'
        if not self.face.enrolled:
            msg += ' Još nijedna osoba nije naučena - upiši ime i pritisni "Nauči lice".'
        if not self.running and self.F.get('watch_arm') is not None:
            self._run_idle('watch', self._go_watch_pose)
        return msg

    def _go_watch_pose(self):
        evt = self._idleAbort
        self._move(2, self.F['watch_arm'], evt=evt)
        self._move(3, self.F['watch_wrist'], evt=evt)

    def enrollFace(self, name=None, pid=None):
        if self.face is None:
            return 'Prepoznavanje lica nije dostupno: %s' % self.faceError
        if self.running:
            return 'Zaustavi berbu pre učenja lica.'
        ok, msg = self.face.start_enroll(name=name, pid=pid)
        if ok:
            self.faceService.poke()
            self._event('info', msg)
        return msg

    def cancelEnroll(self):
        if self.face is None:
            return ''
        return self.face.cancel_enroll()

    def _face_op(self, fn, *args):
        if self.face is None:
            return 'Prepoznavanje lica nije dostupno.'
        ok, msg = fn(*args)
        if ok:
            self._event('info', msg)
        return msg

    def renamePerson(self, pid, name):
        return self._face_op(self.face.rename if self.face else None, pid, name)

    def setPersonWave(self, pid, on):
        return self._face_op(self.face.set_wave if self.face else None, pid, on)

    def deletePerson(self, pid):
        return self._face_op(self.face.delete if self.face else None, pid)

    def forgetFace(self):
        if self.face is None:
            return 'Prepoznavanje lica nije dostupno.'
        msg = self.face.forget()
        self._greeted = {}
        self._event('info', msg)
        return msg

    def testWave(self):
        if self.running:
            return 'Zaustavi berbu pre mahanja.'
        if not self._start_wave(None):
            return 'Robot je upravo zauzet drugim pokretom.'
        return 'Robot maše.'

    def _on_people(self, res):
        """Poziva FaceService kad je bar jedna osoba potvrdjena."""
        if self.running or (self.faceService is not None and self.faceService.suspended):
            return
        now = time.time()
        greet_cd = float(self.F.get('greet_cooldown_s', 30))
        for p in res['people']:
            if now - self._greeted.get(p['pid'], 0) >= greet_cd:
                self._greeted[p['pid']] = now
                self.greeting = {'pid': p['pid'], 'name': p['name'], 'sim': p['sim'], 't': now}
                self._event('greet', 'Prepoznao sam: %s (%d%%)' % (p['name'], int(100 * p['sim'])))
        if not self.F.get('wave_enabled'):
            return
        if now - self._last_wave < float(self.F.get('wave_cooldown_s', 20)):
            return
        wavers = [p for p in res['people'] if p.get('wave')]
        if wavers and self._start_wave(res, wavers[0]['pid']):
            self._event('info', 'Mašem osobi %s.' % wavers[0]['name'])

    def _on_owner(self, res):     # (staro ime - kompatibilnost)
        self._on_people(res)

    def _start_wave(self, res, pid=None):
        ok = self._run_idle('wave', self._wave, res, pid)
        if ok:
            self._last_wave = time.time()
        return ok

    def _wave(self, res, pid=None):
        """Mahanje: (okreni se ka osobi) -> malo podigni ruku -> pan levo-
        desno nekoliko puta (hvataljka se otvara/zatvara kao dlan) -> nazad u
        polozaj u kom je robot bio. Prekida se odmah ako se pokrene berba."""
        F, P, evt = self.F, self.P, self._idleAbort
        try:
            home = {ch: self._cur(ch) for ch in (1, 2, 3, 4)}
            center = home[1]
            if res is not None and F.get('turn_to_face', True) and self.face is not None:
                box = self.face.owner_box(res, pid)
                if box is not None:
                    cx = box[0] + box[2] / 2.0
                    deg = (cx - res['w'] / 2.0) * P['hfov_deg'] / float(res['w'])
                    center = _clamp(home[1] - P['pan_sign'] * deg, -P['pan_limit'], P['pan_limit'])
            sp = float(F.get('wave_speed_dps', 90))
            amp = float(F.get('wave_amplitude', 25))
            lim = P['pan_limit']
            center = _clamp(center, -lim + amp, lim - amp)
            if not self._move(1, center, sp, evt=evt):
                return
            lift = float(F.get('wave_arm_lift', 20))
            if lift:
                a = home[2]
                up = a - lift if abs(a) < 5 else a - math.copysign(lift, a)
                if not self._move(2, _clamp(up, -90, 90), 45, evt=evt):
                    return
            claw_on = bool(F.get('wave_claw', True)) and not self._holding_tomato()   # ne ispusti paradajz!
            half = (P['claw_open'] + P['claw_closed']) / 2.0
            for i in range(int(F.get('wave_cycles', 3))):
                for side in (1, -1):
                    if claw_on:
                        self._servo(4, int(P['claw_open'] if side > 0 else half))
                        self._ang[4] = P['claw_open'] if side > 0 else half
                    if not self._move(1, center + side * amp, sp, evt=evt):
                        return
            if not self._move(1, center, sp, evt=evt):
                return
            self.waveCount += 1
            self._move(4, home[4], evt=evt)
            self._move(2, home[2], 30, evt=evt)
            self._move(1, home[1], sp * 0.6, evt=evt)
        except Exception as e:
            self._error('Mahanje: %s' % e)
        finally:
            self._last_wave = time.time()

    # ==================================================================
    # Status za stranicu
    # ==================================================================
    def events_since(self, last_id):
        try:
            last_id = int(last_id)
        except (TypeError, ValueError):
            last_id = 0
        if last_id > self._event_id:     # server je restartovan
            last_id = 0
        return [e for e in list(self._events) if e['id'] > last_id]

    def _brightness(self):
        """Srednja osvetljenost slike (0-255), racuna se najvise 1x u 2 s."""
        t, v = self._bright
        now = time.time()
        if now - t < 2.0:
            return v
        f = Camera.last_raw
        val = None
        if f is not None and now - Camera.last_raw_t < 2.0:
            try:
                val = float(cv2.cvtColor(f[::8, ::8], cv2.COLOR_BGR2GRAY).mean())
            except Exception:
                val = None
        self._bright = (now, val)
        return val

    def getStatus(self, events_after=None):
        det = self._det()
        r = det.latest() or {}
        b = self._bucket().latest() or {}
        W = r.get('w') or CVThread.videoW
        H = r.get('h') or CVThread.videoH
        P, F, U = self.P, self.F, self.U
        now = time.time()
        stale = (now - r['t']) > 1.0 if r else True
        telemetry = {
            'detected': bool(r.get('detected')) and not stale,
            'tracking': bool(r.get('tracking')),
            'boxX': r.get('x'), 'boxY': r.get('y'), 'radius': r.get('r'),
            'conf': r.get('conf', 0.0), 'prob': r.get('prob', 0.0),
            'candidates': len(r.get('candidates', [])),
            'fps': r.get('fps', 0.0),
            'videoW': W, 'videoH': H,
            'targetRadius': P['grab_r_frac'] * W,
            'grabX': P['grab_x_frac'] * W, 'grabY': P['grab_y_frac'] * H,
            'pan': round(self.pan, 1), 'steer': self.steer,
            'xLock': bool(r.get('detected')) and r.get('x') is not None and
                     abs(r['x'] - P['grab_x_frac'] * W) < P['center_tol_frac'] * W,
            'yLock': bool(r.get('detected')) and r.get('r') is not None and
                     r['r'] >= P['grab_r_frac'] * W * (1 - P['grab_tol_frac']),
            'cameraStale': stale,
            'modelOk': det.model.ready,
            'mask': Camera.tomato_show_mask,
            'logEnabled': Camera.datalog.enabled,
            'logCount': Camera.datalog.count,
            'logError': Camera.datalog.last_error,
            'brightness': self._brightness(),
            'ultraOk': self.ultraOk,
            'color': {k: det.cfg[k] for k in ('hue_center', 'hue_halfwidth', 's_min', 'v_min')},
        }
        bucket = {
            'detected': bool(b.get('detected')) and now - b.get('t', 0) < 1.0,
            'x': b.get('x'), 'w': b.get('bw'), 'imgW': b.get('w_img'),
            'widthPct': (100.0 * b['bw'] / b['w_img']) if b.get('bw') else None,
            'targetPct': 100.0 * P['bucket_near_w_frac'],
            'occludedPct': round(100.0 * (b.get('occluded') or 0.0)),
            'enabled': P.get('deposit_mode', 'bucket') == 'bucket',
            'target': self._deliver_target_quiet(),
            'color': {k: self._bucket().cfg[k] for k in ('hue_center', 'hue_halfwidth', 's_min', 'v_min')},
        }
        fr = self.face
        fs = self.faceService
        fres = fr.latest() if fr is not None else None
        fresh = bool(fres and now - fres['t'] < 1.5)
        people = fr.people_list() if fr is not None else []
        face = {
            'available': bool(fr is not None and fr.available),
            'reason': self.faceError if (fr is None or not fr.available) else '',
            'detector': fr.detector_kind if fr is not None else None,
            'recognize': bool(F.get('recognize')),
            'enabled': bool(F.get('wave_enabled')),
            'enrolled': bool(fr is not None and fr.enrolled),
            'people': people,
            'peopleCount': len(people),
            'enrolling': bool(fr is not None and fr.enrolling),
            'enrollName': fr.enroll_name if fr is not None else '',
            'progress': list(fr.enroll_progress) if fr is not None else [0, 0],
            'coverage': fr.enroll_coverage if fr is not None else {},
            'msg': fr.enroll_msg if fr is not None else '',
            'result': list(fr.enroll_result) if fr is not None and fr.enroll_result else None,
            'faces': len(fres['faces']) if fresh else 0,
            'recognized': [{'pid': p['pid'], 'name': p['name'], 'sim': p['sim']} for p in fres['people']] if fresh else [],
            'owner': bool(fresh and fres['people']),
            'best': fres.get('best') if fresh else None,
            'threshold': round(fr.threshold, 2) if fr is not None else None,
            'greeting': self.greeting,
            'ms': round(fs.last_ms) if fs is not None else 0,
            'paused': bool(fs is not None and fs.suspended),
            'waving': self.waving,
            'waves': self.waveCount,
            'cooldown': max(0, round(float(F.get('wave_cooldown_s', 20)) - (now - self._last_wave))),
            'watchPose': F.get('watch_arm') is not None,
        }
        run_time = self._run_total + ((now - self._run_started) if self._run_started else 0.0)
        calib = {
            'grab': bool(P.get('grab_calibrated')),
            'bucket': bool(P.get('bucket_calibrated')),
            'drop': P.get('drop_arm') is not None,
            'place': bool(P.get('place_calibrated')),
            'clawMask': sorted(self._claw_masks.keys()),
            'look': True,
            'claw': [P['claw_open'], P['claw_closed']],
            'grabPose': [P['grab_arm'], P['grab_wrist']],
            'lookPose': [P['look_arm'], P['look_wrist']],
            'dropPose': [P.get('drop_arm'), P.get('drop_wrist')],
            'carryPose': [P.get('carry_arm'), P.get('carry_wrist')],
            'placePose': [P.get('place_pan'), P.get('place_arm'), P.get('place_wrist')],
            'depositMode': P.get('deposit_mode', 'bucket'),
        }
        stats = dict(self.stats)
        stats.update({'picked': self.pickedCount, 'bucket': self.bucketCount})
        out = {
            'running': self.running,
            'episode': self.episode if self.running else None,
            'state': self.state,
            'stateFor': round(now - self._phase_start, 1) if self._phase_start else 0,
            'busy': self.idleBusy,
            'pickedCount': self.pickedCount,
            'bucketCount': self.bucketCount,
            'attempts': self.attempts,
            'failedGrabs': self.failedGrabs,
            'runTime': round(run_time, 1),
            'avgCycle': round(sum(self.cycleTimes) / len(self.cycleTimes), 1) if self.cycleTimes else None,
            'lastCycle': round(self.cycleTimes[-1], 1) if self.cycleTimes else None,
            'lastError': self.lastError,
            'lastInfo': self.lastInfo,
            'telemetry': telemetry,
            'bucket': bucket,
            'face': face,
            'calib': calib,
            'stats': stats,
            'ui': {'controlMode': U.get('control_mode', 'auto'), 'assistant': bool(U.get('assistant', True)),
                   'voice': bool(U.get('voice', False))},
            'servo': {str(ch): self._current_angle(ch) for ch in range(5)},
            'eventId': self._event_id,
            'bootId': self.bootId,
        }
        if U.get('assistant', True):
            try:
                out['assistant'] = assistant.analyze(out, P, F, {
                    'brightness': telemetry['brightness'], 'ultra_ok': self.ultraOk,
                    'claw_mask': len(self._claw_masks) == 2})
            except Exception as e:
                out['assistant'] = [{'id': 'err', 'level': 'info', 'title': 'Asistent: greška',
                                     'text': str(e), 'action': None}]
        if events_after is not None:
            out['events'] = self.events_since(events_after)
        return out

    def _deliver_target_quiet(self):
        if self.P.get('deposit_mode', 'bucket') != 'bucket' or not self.P.get('bucket_calibrated'):
            return 'pose'
        return 'bucket'

    # ==================================================================
    # Faze berbe
    # ==================================================================
    def _phaseSearch(self):
        P = self.P
        self._stop_motors()
        r = self._fresh(self._t_cmd + P['settle_s'])
        if self._stopRequested.is_set():
            return
        self._log(r)
        if r is None:
            self._error('Kamera ne šalje slike (otvori video na stranici).')
            self._sleep(0.3)
            return
        if self.lastError.startswith('Kamera ne šalje'):
            self.lastError = ''
        if r['detected']:
            self._last_seen = time.time()
            self._set_state(self.STATE_CENTER, 'Paradajz pronađen.')
            return
        if r['tracking']:
            return   # kandidat se upravo prati - NE pomeraj pogled dok se ne potvrdi
        self._sweep_step()

    def _sweep_step(self):
        """Jedan korak okretanja pogleda levo-desno. Vraca True na kraju luka."""
        P = self.P
        nxt = self.pan + P['search_step'] * self._searchDir
        edge = False
        if abs(nxt) >= P['search_sweep']:
            nxt = math.copysign(P['search_sweep'], nxt)
            self._searchDir = -self._searchDir
            self._set_pan(nxt)
            self._sleep(P['search_edge_pause'])
            edge = True
        else:
            self._set_pan(nxt)
        self._sleep(P['search_step_delay'])
        return edge

    def _pan_correct(self, r, gain_scale=1.0, target_x=None, x=None):
        """P-regulator pan servoa ka zadatom x (podrazumevano sredina kadra).
        Vraca gresku u px."""
        P = self.P
        w = r.get('w') or r.get('w_img')
        gx = (0.5 * w) if target_x is None else target_x
        err = (r['x'] if x is None else x) - gx
        if abs(err) > P['center_tol_frac'] * w * 0.6:
            deg = err * P['hfov_deg'] / w
            step = _clamp(-P['pan_sign'] * P['pan_kp'] * gain_scale * deg,
                          -P['pan_max_step'], P['pan_max_step'])
            self._set_pan(self.pan + step)
        return err

    def _lost(self, where):
        self._stop_motors()
        if time.time() - self._last_seen > self.P['lost_timeout_s']:
            self.stats['lostTomato'] += 1
            self._det().tracker.reset()
            self._set_state(self.STATE_SEARCH, 'Meta izgubljena u fazi %s - nastavljam traženje.' % where)

    def _phaseCenter(self):
        P = self.P
        self._stop_motors()
        r = self._fresh(self._t_cmd + P['settle_s'])
        if self._stopRequested.is_set():
            return
        self._log(r)
        if r is None or not r['detected']:
            self._lost('CENTRIRANJE')
            return
        self._last_seen = time.time()
        err = self._pan_correct(r)
        self._set_steer(P['steer_sign'] * P['steer_gain'] * self.pan)
        if abs(err) <= P['center_tol_frac'] * r['w']:
            self._approach_start = time.time()
            self.attempts += 1
            self._grab_tries = 0
            self._fine_backups = 0
            self._set_state(self.STATE_APPROACH, 'Meta centrirana - prilazim.')

    def _pulse(self, speed, dur, direction=1):
        if self._stopRequested.is_set():
            return False
        move.move(speed, direction, 'mid')
        ok = self._sleep(dur)
        self._stop_motors()
        self._t_cmd = time.time()
        return ok and self._sleep(self.P['pulse_off_s'])

    def _phaseApproach(self):
        """Prilazak u pozi gledanja dok paradajz ne bude 'blizu'."""
        P = self.P
        r = self._fresh(self._t_cmd + P['settle_s'])
        if self._stopRequested.is_set():
            self._stop_motors()
            return
        self._log(r)
        if r is None:
            self._stop_motors()
            self._error('Slika sa kamere kasni - stajem.')
            self._lost('PRILAZAK')
            return
        if not r['detected']:
            self._lost('PRILAZAK')
            return
        self._last_seen = time.time()
        if time.time() - self._approach_start > P['max_approach_s']:
            self._stop_motors()
            self.stats['approachTimeouts'] += 1
            self._set_state(self.STATE_SEARCH, 'Prilazak predugo traje - ponovo tražim.')
            return
        err = self._pan_correct(r, 0.8)
        self._set_steer(P['steer_sign'] * P['steer_gain'] * self.pan)
        near_r = P['near_r_frac'] * r['w']
        near = r['r'] >= near_r or r['y'] >= P['near_y_frac'] * r['h']
        dist = self._ultra() if P.get('use_ultra', True) else None
        if dist is not None and dist < P['ultra_stop_cm']:
            near = True
        centered = abs(err) <= P['center_tol_frac'] * r['w'] * 1.5
        if near and centered:
            self._stop_motors()
            self._set_state(self.STATE_LOWER, 'Blizu sam - polako spuštam ruku i otvaram hvataljku.')
            return
        if near:
            return   # samo poravnaj (pan/tockovi), bez voznje
        frac = _clamp((near_r - r['r']) / max(near_r, 1.0) * 1.5, 0.0, 1.0)
        dur = max(P['min_pulse_s'], P['pulse_on_s'] * frac)
        self._pulse(P['approach_speed'], dur)

    def _phaseLower(self):
        """Glatko spustanje u pozu hvatanja; od sad se paradajz gleda iz nje."""
        P = self.P
        self._stop_motors()
        self._set_steer(P['steer_sign'] * P['steer_gain'] * self.pan)
        if not self._grab_pose():
            return
        self._sleep(P['servo_settle_s'])
        self._det().tracker.reset()
        self._fine_start = time.time()
        self._last_seen = time.time()
        self._t_cmd = time.time()
        self._set_state(self.STATE_FINE, 'Navodim paradajz između prstiju hvataljke.')

    def _abort_to_search(self, why):
        self._stop_motors()
        if self.episode not in ('full', 'one'):
            self._finish_episode(why, 'warn')
            return
        self._look_pose()
        self._det().tracker.reset()
        self._t_cmd = time.time()
        self._set_state(self.STATE_SEARCH, why)

    def _phaseFine(self):
        """Fini prilazak u pozi hvatanja do kalibrisanog mesta u kadru."""
        P = self.P
        r = self._fresh(self._t_cmd + P['settle_s'])
        if self._stopRequested.is_set():
            self._stop_motors()
            return
        self._log(r)
        vis = r is not None and r.get('x') is not None and (r['detected'] or r['tracking'])
        if not vis:
            self._stop_motors()
            if time.time() - self._last_seen > P['fine_lost_s']:
                if self._fine_backups < P['max_grab_attempts']:
                    self._fine_backups += 1
                    self._pulse(P['fine_speed'], P['backup_s'], -1)
                    self._last_seen = time.time()
                    self._info('Ne vidim paradajz iz poze hvatanja - malo unazad (%d).' % self._fine_backups)
                else:
                    self.stats['lostTomato'] += 1
                    self._abort_to_search('Paradajz izgubljen pri navođenju - tražim ponovo.')
            return
        self._last_seen = time.time()
        if time.time() - self._fine_start > P['max_fine_s']:
            self._abort_to_search('Navođenje predugo traje - tražim ponovo.')
            return
        gx, gy, gr = self._grab_point(r)
        tol = P['grab_tol_frac']
        err_x = self._pan_correct(r, 0.6, target_x=gx)
        self._set_steer(P['steer_sign'] * P['steer_gain'] * self.pan)
        ok_x = abs(err_x) <= tol * r['w']
        ok_depth = r['r'] >= gr * (1.0 - tol * 2.5) or r['y'] >= gy - tol * r['h']
        too_close = r['r'] > gr * 1.35 or r['y'] > gy + 2.5 * tol * r['h']
        if ok_x and ok_depth and not too_close:
            self._stop_motors()
            self._set_state(self.STATE_GRAB, 'Paradajz je između prstiju - hvatam.')
            return
        if too_close:
            self._pulse(P['fine_speed'], P['min_pulse_s'], -1)   # preblizu: malo unazad
            return
        if not ok_x:
            return   # samo poravnavanje (pan/tockovi)
        self._pulse(P['fine_speed'], P['fine_pulse_s'])

    def _grip_ok(self):
        """Posto je kamera na hvataljci, uhvacen paradajz ostaje na (priblizno)
        istom mestu u kadru i posle podizanja ruke. Ako ga tamo nema - nije uhvacen."""
        P = self.P
        if not P.get('grip_verify', True):
            return True
        hits = seen = 0
        t0 = time.time()
        after = time.time()
        while time.time() - t0 < 1.2 and not self._stopRequested.is_set():
            r = self._fresh(after, timeout=0.5)
            if r is None:
                continue
            after = r['t']
            seen += 1
            gx, gy, gr = self._grab_point(r)
            for c in self._det().last_candidates():
                if (math.hypot(c.x - gx, c.y - gy) < max(2.0 * gr, 0.18 * r['w'])
                        and 0.45 * gr < c.r < 2.2 * gr and c.prob > 0.25):
                    hits += 1
                    break
        self._log(force=True)
        if self._stopRequested.is_set():
            return False
        if seen == 0:
            return True   # nema slike - ne mozemo proveriti, nastavi
        return hits >= max(1, seen // 3)

    def _lift_angle(self):
        """Ugao ruke za podizanje sa paradajzom: 'lift_delta' stepeni od
        TRENUTNOG ugla ruke u smeru ka pozi gledanja (pozi gledanja je ruka
        uvek iznad poze hvatanja - radi bez obzira na smer montaze servoa i
        kad je ruka rucno pomerena). Ako je 'lift_arm_abs' zadat, koristi se on."""
        P = self.P
        if P.get('lift_arm_abs') is not None:
            return P['lift_arm_abs']
        g = float(self._cur(2))          # STVARNI ugao ruke (i posle rucnog pomeranja)
        d = float(P.get('lift_delta', 35))
        look, grab = P.get('look_arm'), P.get('grab_arm')
        if look is not None and grab is not None and abs(float(look) - float(grab)) >= 5:
            up = 1.0 if float(look) > float(grab) else -1.0   # "gore" = smer ka pozi gledanja
            return _clamp(g + up * d, -90, 90)
        if abs(g) < 5:
            return g - d
        return _clamp(g - math.copysign(d, g), -90, 90)

    def _close_claw(self):
        """Postepeno zatvaranje hvataljke - paradajz se ravnomerno stegne, a
        servo ne povuce naglo veliku struju."""
        P = self.P
        if not self._move(4, P['claw_closed'], P.get('claw_close_dps', 30)):
            return False
        return self._sleep(P.get('claw_hold_s', 0.6))

    def _phaseGrab(self):
        P = self.P
        self._stop_motors()
        manual = self.episode == 'grab'
        if P['creep_s'] > 0 and not manual:
            if not self._pulse(P['fine_speed'], P['creep_s']):
                return
        if manual and self._cur(4) < P['claw_open'] - 5:
            if not self._move(4, P['claw_open']):      # rucno: prvo otvori ako nije
                return
        if not self._sleep(P.get('pause_before_grab_s', 0.5)):
            return
        self.stats['grabAttempts'] += 1
        if not self._close_claw():
            return
        if not self._move(2, self._lift_angle(), P.get('arm_lift_dps', 20)):
            return
        self._sleep(0.3)
        grip = self._grip_ok()
        if self._stopRequested.is_set():
            return                      # STOP nije neuspelo hvatanje
        if grip:
            self.holding = True
            if manual:
                self._finish_episode('Paradajz uhvaćen - drži ga. Sledeće: "Ubaci u kanticu" ili "Odloži".')
            elif self._deliver_target() == 'bucket':
                self._begin_bucket_search('Paradajz uhvaćen - tražim kanticu.')
            else:
                self._set_state(self.STATE_DEPOSIT, 'Paradajz uhvaćen - odlažem ga na pozu odlaganja.')
            return
        self._grab_tries += 1
        self.failedGrabs += 1
        self.stats['grabFails'] += 1
        self._error('Hvatanje nije uspelo (pokušaj %d).' % self._grab_tries)
        self._move(4, P['claw_open'])
        if self._grab_tries >= P['max_grab_attempts']:
            self._abort_to_search('Hvatanje nije uspelo - tražim ponovo.' if not manual else
                                  'Hvatanje nije uspelo - namesti ruku tako da je paradajz između prstiju i probaj ponovo.')
            return
        self._pulse(P['fine_speed'], P['backup_s'], -1)
        self._set_state(self.STATE_LOWER, 'Ponovni pokušaj hvatanja.')

    # ---------------- kantica ----------------
    def _begin_bucket_search(self, info, carry=True):
        self._bucket().reset()
        self._sweeps = 0
        self._bucket_max_w = 0.0
        self._bucket_search_start = time.time()
        self._set_state(self.STATE_FIND_BUCKET, info)
        if carry:
            self._carry_pose()
        self._t_cmd = time.time()

    def _phaseFindBucket(self):
        P = self.P
        self._stop_motors()
        if time.time() - self._bucket_search_start > P.get('bucket_search_timeout_s', 60):
            self.stats['bucketTimeouts'] += 1
            self._error('Kantica nije pronađena - odlažem paradajz na pozu odlaganja.')
            self._set_state(self.STATE_DEPOSIT, 'Kantica nije pronađena - odlažem na pozu odlaganja.')
            return
        b = self._fresh(self._t_cmd + P['settle_s'], source=self._bucket())
        if self._stopRequested.is_set():
            return
        self._log()
        if b is None:
            self._error('Kamera ne šalje slike (otvori video na stranici).')
            self._sleep(0.3)
            return
        if self.lastError.startswith('Kamera ne šalje'):
            self.lastError = ''
        if b['detected']:
            self._last_seen = time.time()
            self._approach_start = time.time()
            self._bucket_max_w = b['bw'] or 0.0
            self._set_state(self.STATE_BUCKET_APPROACH, 'Kantica pronađena - prilazim.')
            return
        if b['x'] is not None and time.time() - self._phase_start < 30:
            return   # kandidat se proverava - ne pomeraj pogled
        if self._sweep_step():
            if self._stopRequested.is_set():
                return
            self._sweeps += 1
            if self._sweeps % 2 == 0:
                # ceo luk bez kantice: kratka voznja u luku da se promeni pogled
                self._info('Kantica nije u vidokrugu - menjam položaj.')
                self._set_steer(P['steer_max'] * (1 if (self._sweeps // 2) % 2 else -1))
                self._sleep(0.15)
                self._pulse(P.get('bucket_approach_speed', 30), P.get('bucket_turn_s', 0.7))
                self._set_steer(0)

    def _phaseBucketApproach(self):
        P = self.P
        b = self._fresh(self._t_cmd + P['settle_s'], source=self._bucket())
        if self._stopRequested.is_set():
            self._stop_motors()
            return
        self._log()
        W = float((b or {}).get('w_img') or CVThread.videoW or 320)
        target_w = P.get('bucket_near_w_frac', 0.45) * W
        if b is None or not b['detected']:
            self._stop_motors()
            since = time.time() - self._last_seen
            # kantica je NESTALA IZBLIZA (prekrila kadar / ispod kamere) -> stigli smo
            if self._bucket_max_w >= P.get('bucket_arrive_frac', 0.75) * target_w and since < 3.0:
                self._set_state(self.STATE_DROP, 'Kantica je izbliza izašla iz kadra - stigao sam, ubacujem.')
                return
            if since > P.get('bucket_lost_s', 2.0):
                self.stats['bucketLost'] += 1
                self._set_state(self.STATE_FIND_BUCKET, 'Kantica izgubljena iz vida - tražim je ponovo.')
                self._t_cmd = time.time()
            return
        self._last_seen = time.time()
        self._bucket_max_w = max(self._bucket_max_w, b['bw'])
        if time.time() - self._approach_start > P.get('bucket_max_approach_s', 45):
            self._stop_motors()
            self.stats['bucketTimeouts'] += 1
            self._set_state(self.STATE_DEPOSIT, 'Prilazak kantici predugo traje - odlažem na pozu odlaganja.')
            return
        tx = P.get('bucket_x_frac', 0.5) * W
        err = self._pan_correct({'w': W}, 0.7, target_x=tx, x=b['x'])
        self._set_steer(P['steer_sign'] * P['steer_gain'] * self.pan)
        dist = self._ultra() if P.get('use_ultra', True) else None
        near = b['bw'] >= 0.92 * target_w or (dist is not None and dist < P.get('bucket_ultra_cm', 12))
        too_close = b['bw'] > 1.35 * target_w and b.get('borders', 0) < 3
        centered = abs(err) <= P['center_tol_frac'] * W * 1.6
        if near and centered and not too_close:
            self._stop_motors()
            self._set_state(self.STATE_DROP, 'Stigao sam do kantice (širina %d%% kadra) - ubacujem.' % int(100 * b['bw'] / W))
            return
        if too_close:
            self.stats['bucketBackups'] += 1
            self._pulse(P.get('bucket_approach_speed', 30), P['min_pulse_s'], -1)
            return
        if near:
            return   # samo poravnaj
        frac = _clamp((target_w - b['bw']) / max(target_w, 1.0) * 1.5, 0.0, 1.0)
        dur = max(P['min_pulse_s'], P['pulse_on_s'] * frac)
        self._pulse(P.get('bucket_approach_speed', 30), dur)

    def _phaseDrop(self):
        """Ruka iznad otvora kantice -> polako otvori -> malo unazad -> nazad."""
        P = self.P
        self._stop_motors()
        arm = self._lift_angle() if P.get('drop_arm') is None else P['drop_arm']
        wrist = self._cur(3) if P.get('drop_wrist') is None else P['drop_wrist']
        if not self._move(3, wrist, P.get('wrist_speed_dps', 20)):
            return
        if not self._move(2, arm, P.get('carry_speed_dps', 15)):
            return
        if not self._sleep(P.get('pause_after_place_s', 0.8)):
            return
        if not self._move(4, P['claw_open'], P.get('claw_release_dps', 12)):
            return
        self.holding = False
        if not self._sleep(P.get('pause_after_release_s', 0.8)):
            return
        self.pickedCount += 1
        self.bucketCount += 1
        self._finish_cycle()
        # PRVO malo unazad (ruka je jos iznad ivice kantice), tek onda ruka dole
        if P.get('drop_backup_s', 0.6) > 0:
            self._set_steer(0)
            if not self._pulse(P.get('bucket_approach_speed', 30), P['drop_backup_s'], -1):
                return
        self._back_to_search('Paradajz ubačen u kanticu (#%d).' % self.bucketCount)

    def _phaseDeposit(self):
        """Odlaganje na snimljenu POZU ODLAGANJA (okret, ruka, zglob)."""
        P = self.P
        self._stop_motors()
        self._set_steer(0)
        pan = P.get('place_pan')
        pan = P['basket_pan'] if pan is None else pan
        arm = P.get('place_arm')
        if arm is None:
            arm = P['grab_arm'] if P.get('basket_arm') is None else P['basket_arm']
        wrist = P.get('place_wrist')
        if wrist is None:
            wrist = P.get('basket_wrist', 0)
        if not self._move(1, pan, P.get('carry_speed_dps', 15)):
            return
        self._sleep(0.3)
        if not self._move(3, wrist, P.get('wrist_speed_dps', 20)):
            return
        if not self._move(2, arm, P.get('place_down_dps', 10)):
            return
        if not self._sleep(P.get('pause_after_place_s', 0.8)):
            return
        if not self._move(4, P['claw_open'], P.get('claw_release_dps', 12)):
            return
        self.holding = False
        if not self._sleep(P.get('pause_after_release_s', 0.8)):
            return
        if not self._move(2, self._lift_angle(), P.get('arm_speed_dps', 25)):
            return
        self.pickedCount += 1
        self._finish_cycle()
        self._back_to_search('Paradajz odložen (#%d).' % self.pickedCount)

    def _finish_cycle(self):
        now = time.time()
        if self._cycle_start is not None:
            self.cycleTimes.append(now - self._cycle_start)
        self._cycle_start = now

    def _back_to_search(self, info):
        if self.episode in ('one', 'deliver', 'place'):
            self.pan = 0.0
            self._look_pose()
            self._finish_episode(info)
            return
        self.pan = 0.0
        self._look_pose()
        self._det().tracker.reset()
        self._sleep(self.P['servo_settle_s'])
        self._t_cmd = time.time()
        self._set_state(self.STATE_SEARCH, info)

    # ==================================================================
    def run(self):
        phases = {
            self.STATE_SEARCH: self._phaseSearch,
            self.STATE_CENTER: self._phaseCenter,
            self.STATE_APPROACH: self._phaseApproach,
            self.STATE_LOWER: self._phaseLower,
            self.STATE_FINE: self._phaseFine,
            self.STATE_GRAB: self._phaseGrab,
            self.STATE_FIND_BUCKET: self._phaseFindBucket,
            self.STATE_BUCKET_APPROACH: self._phaseBucketApproach,
            self.STATE_DROP: self._phaseDrop,
            self.STATE_DEPOSIT: self._phaseDeposit,
        }
        while True:
            self._wake.wait()
            if self._stopRequested.is_set() or not self.running:
                self._wake.clear()
                time.sleep(0.05)
                continue
            try:
                if self._need_pose:
                    self._need_pose = False
                    self._look_pose()
                    self._sleep(self.P['servo_settle_s'])
                fn = phases.get(self.state)
                if fn is None:
                    self._wake.clear()
                else:
                    self._busy = True
                    try:
                        fn()
                    finally:
                        self._busy = False
            except Exception as e:
                self._error('Interna greška: %s' % e)
                print('[TOMATO-PICK] GRESKA: %s' % e, flush=True)
                self._stop_motors()
                time.sleep(0.5)


if __name__ == '__main__':
    print('Ovaj modul se pokreće iz WebServer.py, ne samostalno.')
