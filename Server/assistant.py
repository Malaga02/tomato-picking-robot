#!/usr/bin/env python3
# File name   : assistant.py
# Description : "AI asistent" - dijagnostika robota i preporuke u realnom vremenu
# Author      : napisano za master rad Adnana Malagica (dopuna Adeept PiCar-Pro softvera)
#
# Ekspertski sistem (pravila zasnovana na znanju o robotu) koji iz telemetrije
# (stanje kamere, detekcija, statistika hvatanja, kantica, kalibracije, lica,
# osvetljenje) zakljucuje sta ne valja ili sta moze bolje, i za svaki nalaz
# predlaze konkretnu akciju (dugme "Primeni" na stranici). Radi lokalno na
# robotu - ne treba internet. Asistent se moze iskljuciti; tada robot radi
# potpuno isto, samo bez preporuka (rucni rad).
#
# Svaki nalaz: {'id', 'level': 'crit'|'warn'|'info'|'ok', 'title', 'text',
#               'action': None | {'label', 'cmd'} | {'label', 'set': {...}} |
#                                {'label', 'tab'}}

LEVEL_ORDER = {'crit': 0, 'warn': 1, 'info': 2, 'ok': 3}


def _pct(a, b):
    return 100.0 * a / b if b else 0.0


def analyze(st, P, F, extra=None):
    """st: status (TomatoPicker.getStatus bez asistenta), P: picker podesavanja,
    F: podesavanja lica, extra: {'brightness', 'ultra_ok', 'claw_mask'}."""
    extra = extra or {}
    out = []
    add = out.append
    tel = st.get('telemetry', {})
    bucket = st.get('bucket', {})
    face = st.get('face', {})
    calib = st.get('calib', {})
    stats = st.get('stats', {})
    running = st.get('running')
    bucket_mode = P.get('deposit_mode', 'bucket') == 'bucket'

    # ---- kamera ----
    if tel.get('cameraStale'):
        add({'id': 'camera', 'level': 'crit', 'title': 'Kamera ne šalje slike',
             'text': 'Robot vidi samo dok je video otvoren na stranici. Osveži stranicu; ako ne pomogne, '
                     'proveri USB kabl kamere.', 'action': None})
    elif tel.get('fps') and tel['fps'] < 5:
        add({'id': 'fps', 'level': 'warn', 'title': 'Spora obrada slike (%.1f fps)' % tel['fps'],
             'text': 'Zatvori druge kartice sa videom robota; prepoznavanje lica troši procesor - '
                     'isključi ga dok robot bere.', 'action': None})
    b = extra.get('brightness')
    if b is not None and not tel.get('cameraStale'):
        if b < 55:
            add({'id': 'dark', 'level': 'warn', 'title': 'Previše tamno (%.0f/255)' % b,
                 'text': 'Pri slabom svetlu boje izblede i paradajz/kantica se teže prepoznaju. '
                         'Upali svetlo ili posle toga ponovo podesi boju dodirom.', 'action': None})
        elif b > 215:
            add({'id': 'bright', 'level': 'warn', 'title': 'Preeksponirana slika (%.0f/255)' % b,
                 'text': 'Direktno sunce/lampa u kameru - boje postaju bele. Okreni robota ili smanji svetlo.',
                 'action': None})

    # ---- kalibracije ----
    if not calib.get('grab'):
        add({'id': 'calib_grab', 'level': 'crit', 'title': 'Hvatanje nije kalibrisano',
             'text': 'Bez ove kalibracije robot ne zna gde je paradajz između prstiju hvataljke.',
             'action': {'label': 'Otvori kalibraciju', 'tab': 'calib'}})
    if bucket_mode and not calib.get('bucket'):
        add({'id': 'calib_bucket', 'level': 'warn', 'title': 'Kantica nije kalibrisana',
             'text': 'Dok se ne kalibriše, robot paradajz ostavlja na pozu odlaganja (ne u kanticu).',
             'action': {'label': 'Kalibriši kanticu', 'tab': 'calib'}})
    if bucket_mode and calib.get('bucket') and not calib.get('drop'):
        add({'id': 'calib_drop', 'level': 'warn', 'title': 'Poza ubacivanja nije snimljena',
             'text': 'Robot tada samo podigne ruku - paradajz može da padne pored kantice. '
                     'Snimi pozu u kojoj je hvataljka tačno iznad otvora.',
             'action': {'label': 'Otvori kalibraciju', 'tab': 'calib'}})
    if not bucket_mode and not calib.get('place'):
        add({'id': 'calib_place', 'level': 'info', 'title': 'Poza odlaganja nije snimljena',
             'text': 'Paradajz se ostavlja levo od robota na pod. Možeš snimiti svoju pozu odlaganja '
                     '(okret, ruka, zglob) jednim dugmetom.', 'action': {'label': 'Otvori kalibraciju', 'tab': 'calib'}})
    if bucket_mode and not extra.get('claw_mask'):
        add({'id': 'claw_mask', 'level': 'info', 'title': 'Maska hvataljke nije kalibrisana',
             'text': 'Prsti hvataljke zaklanjaju donji deo kantice. Robot ih procenjuje automatski, '
                     'ali je kalibracija (robot okrene ruku levo-desno ~6 s) pouzdanija.',
             'action': {'label': 'Kalibriši masku', 'cmd': 'tomatoCalibClawMask'}})

    # ---- hvatanje ----
    att = stats.get('grabAttempts', 0)
    fail = stats.get('grabFails', 0)
    if att >= 3 and fail / float(att) >= 0.5:
        add({'id': 'grab_fail', 'level': 'warn',
             'title': 'Hvatanje često ne uspeva (%d od %d)' % (fail, att),
             'text': 'Najčešći uzrok: kalibracija hvatanja je rađena sa drugačije postavljenim '
                     'paradajzom ili se hvataljka ne zatvara dovoljno. Ponovi kalibraciju hvatanja i '
                     'test hvataljke; predlog: sporije zatvaranje i duži kratak pomak napred.',
             'action': {'label': 'Sporije zatvaranje', 'set': {
                 'picker.claw_close_dps': max(6, int(P.get('claw_close_dps', 15)) - 4)}}})
    if stats.get('lostTomato', 0) >= 4 and stats.get('picked', 0) == 0:
        add({'id': 'tomato_lost', 'level': 'warn', 'title': 'Paradajz se često gubi iz vida',
             'text': 'Robot ga vidi pa izgubi. Podesi boju paradajza dodirom na sliku i smanji brzinu prilaska.',
             'action': {'label': 'Sporiji prilazak', 'set': {
                 'picker.approach_speed': max(18, int(P.get('approach_speed', 32)) - 6)}}})
    if tel.get('candidates', 0) >= 4 and not tel.get('detected'):
        add({'id': 'tomato_noise', 'level': 'info', 'title': 'Mnogo crvenih mrlja, nijedna potvrđena',
             'text': 'U kadru je mnogo crvenkastih predmeta. Uključi „Maska boje" da vidiš šta prolazi '
                     'filter, i dodirom podesi boju na sam paradajz.', 'action': None})

    # ---- kantica ----
    if bucket_mode:
        bl = stats.get('bucketLost', 0)
        if bl >= 3:
            add({'id': 'bucket_lost', 'level': 'warn', 'title': 'Kantica se gubi tokom prilaska (%d×)' % bl,
                 'text': 'Kalibriši masku hvataljke i boju kantice dodirom; proveri da kantica nije na '
                         'zelenoj podlozi (trava/lišće).', 'action': {'label': 'Kalibriši masku', 'cmd': 'tomatoCalibClawMask'}})
        if stats.get('bucketBackups', 0) >= 3:
            add({'id': 'bucket_overshoot', 'level': 'info', 'title': 'Robot često prođe kanticu',
                 'text': 'Dolazi preblizu pa se vraća unazad - smanji brzinu prilaska kantici.',
                 'action': {'label': 'Sporije ka kantici', 'set': {
                     'picker.bucket_approach_speed': max(18, int(P.get('bucket_approach_speed', 30)) - 5)}}})
        if stats.get('bucketTimeouts', 0) >= 2:
            add({'id': 'bucket_timeout', 'level': 'warn', 'title': 'Kantica nije nađena (%d×)' % stats['bucketTimeouts'],
                 'text': 'Stavi kanticu u vidno polje robota (ispred, 30–100 cm) ili isključi kanticu i '
                         'koristi pozu odlaganja.', 'action': {'label': 'Isključi kanticu', 'cmd': 'tomatoBucketOff'}})
    # ---- ultrazvuk ----
    if P.get('use_ultra', True) and extra.get('ultra_ok') is False:
        add({'id': 'ultra', 'level': 'info', 'title': 'Ultrazvučni senzor ne odgovara',
             'text': 'Robot zato ne može da koristi ultrazvučnu kočnicu. Proveri priključak senzora.',
             'action': {'label': 'Isključi ultrazvuk', 'set': {'picker.use_ultra': False}}})
    # ---- lica ----
    if face.get('available') and (face.get('enabled') or face.get('recognize')) and not face.get('peopleCount'):
        add({'id': 'face_empty', 'level': 'info', 'title': 'Nijedna osoba nije naučena',
             'text': 'Upiši ime i pritisni „Nauči lice" da bi robot znao ko je ispred njega.',
             'action': {'label': 'Osobe', 'tab': 'people'}})
    if not face.get('available') and (face.get('enabled') or face.get('recognize')):
        add({'id': 'face_na', 'level': 'warn', 'title': 'Prepoznavanje lica nije dostupno',
             'text': face.get('reason') or '', 'action': None})
    if running and (face.get('enabled') or face.get('recognize')):
        add({'id': 'face_pause', 'level': 'info', 'title': 'Prepoznavanje lica je pauzirano',
             'text': 'Dok robot bere, lica se ne analiziraju (procesor je zauzet slikom za berbu).', 'action': None})
    # ---- uspeh ----
    picked = stats.get('picked', 0)
    if picked >= 1 and att:
        rate = _pct(picked, att)
        if rate >= 70:
            add({'id': 'good', 'level': 'ok', 'title': 'Odlično ide: %d ubrano (%.0f%% uspešnih prilazaka)' % (picked, rate),
                 'text': 'Prosečan ciklus %s.' % ('%.0f s' % st['avgCycle'] if st.get('avgCycle') else '–'),
                 'action': None})
    if not [i for i in out if i['level'] in ('crit', 'warn')]:
        if not [i for i in out if i['id'] == 'good']:
            add({'id': 'ready', 'level': 'ok', 'title': 'Sve izgleda spremno',
                 'text': 'Kamera, kalibracije i podešavanja su u redu.', 'action': None})
    out.sort(key=lambda i: LEVEL_ORDER.get(i['level'], 9))
    return out
