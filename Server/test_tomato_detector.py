#!/usr/bin/env python3
# File name   : test_tomato_detector.py
# Description : Provera detektora paradajza nad slikom (bez pokretanja robota)
#
# Upotreba (na Pi-ju ili na racunaru sa python3 + opencv + numpy):
#   python3 test_tomato_detector.py slika.jpg [slika2.jpg ...]
#   python3 test_tomato_detector.py ../../tomato_data/20260929_181500/*.jpg
# Za svaku sliku ispisuje kandidate sa verovatnocom i snima slika_det.jpg sa
# iscrtanim rezultatom. Korisno za master rad (primeri detekcije) i za brzu
# proveru posle promene modela ili kalibracije boje.

import sys
import time

import cv2

import tomato_config
import tomato_detector


def main(paths):
    cfg = tomato_config.load()
    det = tomato_detector.TomatoDetector(cfg.get('detector'))
    print('Boja:', {k: det.cfg[k] for k in ('hue_center', 'hue_halfwidth', 's_min', 'v_min')})
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print('%s: ne mogu da ucitam' % p)
            continue
        det.tracker.reset()
        t0 = time.time()
        res = None
        for k in range(4):   # tracker trazi nekoliko uzastopnih frejmova za potvrdu
            res = det.process(img, t=t0 + k * 0.07)
        ms = (time.time() - t0) / 4 * 1000
        print('%s: %s  (%.1f ms/frejm)' % (p, 'PARADAJZ (%.0f%%) na x=%.0f y=%.0f r=%.0f' % (
            100 * res['conf'], res['x'], res['y'], res['r']) if res['detected'] else 'nema paradajza', ms))
        for c in res['candidates']:
            print('    kandidat x=%5.0f y=%5.0f r=%4.0f  p=%.3f  kruznost=%.2f' % (c['x'], c['y'], c['r'], c['p'], c['circ']))
        out = p.rsplit('.', 1)[0] + '_det.jpg'
        cv2.imwrite(out, det.draw(img.copy(), res))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__ if __doc__ else 'upotreba: python3 test_tomato_detector.py slika.jpg')
        sys.exit(1)
    main(sys.argv[1:])
