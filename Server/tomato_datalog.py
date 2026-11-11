#!/usr/bin/env python3
# File name   : tomato_datalog.py
# Description : Snimanje frejmova sa kamere robota za dalje treniranje modela
#
# Kada je ukljuceno (dugme na tomato.html), svakih log_interval_s sekundi
# snima SIROVI frejm (bez iscrtanih oznaka) + jednu JSON liniju sa onim sto
# je detektor video (kandidati, verovatnoce, stanje robota). Ti frejmovi se
# kasnije preuzmu (WinSCP), oznace i koriste za treniranje modela na PRAVIM
# slikama sa ove kamere - to je najveci pojedinacni dobitak za tacnost.
# Ogranicenja: najvise log_max_frames frejmova i nikad ako na disku ima
# manje od 500 MB slobodnog prostora.

import json
import os
import shutil
import threading
import time

import cv2

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', 'tomato_data'))
MIN_FREE_BYTES = 500 * 1024 * 1024


class DataLogger(object):
    def __init__(self):
        self.enabled = False
        self.interval = 2.0
        self.max_frames = 800
        self.count = 0
        self._last = 0.0
        self._lock = threading.Lock()
        self.last_error = ''
        self.session_dir = None

    def configure(self, enabled=None, interval=None, max_frames=None):
        if interval is not None:
            self.interval = max(0.3, float(interval))
        if max_frames is not None:
            self.max_frames = int(max_frames)
        if enabled is not None:
            self.enabled = bool(enabled)
            if self.enabled and self.session_dir is None:
                self.session_dir = os.path.join(DATA_DIR, time.strftime('%Y%m%d_%H%M%S'))

    def _count_existing(self):
        n = 0
        if os.path.isdir(DATA_DIR):
            for root, _, files in os.walk(DATA_DIR):
                n += sum(1 for f in files if f.endswith('.jpg'))
        return n

    def maybe_log(self, frame, result, state=''):
        if not self.enabled or frame is None:
            return
        now = time.time()
        if now - self._last < self.interval:
            return
        with self._lock:
            self._last = now
            try:
                if self.count == 0:
                    self.count = self._count_existing()
                if self.count >= self.max_frames:
                    self.last_error = 'dostignut limit od %d frejmova' % self.max_frames
                    return
                os.makedirs(self.session_dir, exist_ok=True)
                if shutil.disk_usage(self.session_dir).free < MIN_FREE_BYTES:
                    self.last_error = 'malo slobodnog prostora na disku'
                    return
                name = '%s_%03d' % (time.strftime('%H%M%S'), int((now % 1) * 1000))
                cv2.imwrite(os.path.join(self.session_dir, name + '.jpg'), frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                rec = {'file': name + '.jpg', 't': now, 'state': state}
                if result:
                    rec.update({k: result.get(k) for k in ('detected', 'x', 'y', 'r', 'conf', 'candidates')})
                with open(os.path.join(self.session_dir, 'log.jsonl'), 'a') as f:
                    f.write(json.dumps(rec) + '\n')
                self.count += 1
                self.last_error = ''
            except Exception as e:
                self.last_error = str(e)
