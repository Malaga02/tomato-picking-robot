#!/usr/bin/env python3
# File name   : rebuild_faces.py
# Description : Vraca naucena lica posle ostecenja SD kartice.
#
# Fajl faces/people.json (imena + otisci lica) je izgubljen pri ostecenju
# kartice, ali su sacuvani isecci lica (faces/people/<id>/NNN.jpg, poravnati
# 112x112). Ovaj program iz njih ponovo racuna otiske (SFace) i pravi novi
# people.json. Imena su izgubljena, pa osobe dobijaju privremena imena
# "Osoba 1", "Osoba 2"... - pravo ime se upisuje na stranici, kartica "Osobe".
#
# Pokretanje (iz foldera Server):  sudo python3 rebuild_faces.py
# Ako people.json vec postoji, nista ne menja (osim uz --force).

import os
import shutil
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import face_module  # noqa: E402

MIN_SAMPLES = 3


def valid_crops(d):
    """Ucitaj ispravne isecke iz foldera osobe (preskace prazne/ostecene)."""
    out = []
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if not name.lower().endswith('.jpg') or not os.path.isfile(p) or os.path.getsize(p) == 0:
            continue
        img = cv2.imread(p)
        if img is None or img.ndim != 3 or img.shape[0] < 50 or img.shape[1] < 50:
            continue
        if img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112))
        out.append(img)
    return out


def main():
    force = '--force' in sys.argv
    rec = face_module.FaceRecognizer()
    if not rec.available:
        print('GRESKA: prepoznavanje lica nije dostupno: %s' % rec.reason)
        return 1
    if rec.people and not force:
        print('people.json vec postoji (%d osoba) - nista se ne menja.' % len(rec.people))
        return 0

    people_dir = os.path.join(rec.data_dir, 'people')
    if not os.path.isdir(people_dir):
        print('Nema foldera faces/people - nema sta da se vrati.')
        return 0

    broken_dir = os.path.join(rec.data_dir, 'people_osteceno')
    people = []
    # id osobe je 'p' + vreme ucenja (heksadecimalno), pa sortiranje po id-u
    # daje redosled kojim su osobe ucene
    for pid in sorted(os.listdir(people_dir)):
        d = os.path.join(people_dir, pid)
        if not os.path.isdir(d):
            continue
        crops = valid_crops(d)
        if len(crops) < MIN_SAMPLES:
            os.makedirs(broken_dir, exist_ok=True)
            shutil.move(d, os.path.join(broken_dir, pid))
            print('  %s: samo %d ispravnih slika - premesteno u faces/people_osteceno' % (pid, len(crops)))
            continue
        E = np.array([rec.embed(c) for c in crops], np.float32)
        # izbaci uzorke koji ne lice na ostale (npr. ostecena slika)
        mean = face_module._norm(E.mean(axis=0))
        sims = E @ mean
        keep = E[sims >= max(0.45, float(np.median(sims)) - 0.2)]
        if len(keep) < MIN_SAMPLES:
            keep = E
        try:
            created = time.strftime('%Y-%m-%d %H:%M', time.localtime(int(pid[1:], 16) / 1000.0))
        except ValueError:
            created = None
        people.append({'id': pid, 'name': 'Osoba %d' % (len(people) + 1), 'wave': True,
                       'created': created, 'emb': keep})
        print('  %s -> Osoba %d (%d uzoraka, naucena %s)' % (pid, len(people), len(keep), created))

    if not people:
        print('Nijedna osoba nije mogla da se vrati - lica treba ponovo nauciti na stranici.')
        return 0

    # provera: razlicite osobe ne smeju biti previse slicne (isto lice dva puta)
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            s = float(np.max(people[i]['emb'] @ people[j]['emb'].T))
            if s > 0.6:
                print('  PAZNJA: %s i %s lice na istu osobu (slicnost %.2f) - '
                      'na stranici jednu obrisi ili im daj isto ime.' % (people[i]['name'], people[j]['name'], s))

    rec.people = people
    rec._save()
    print('Vraceno osoba: %d. Imena upisi na stranici (kartica "Osobe" -> preimenuj).' % len(people))
    return 0


if __name__ == '__main__':
    sys.exit(main())
