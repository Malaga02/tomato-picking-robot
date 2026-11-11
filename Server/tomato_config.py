#!/usr/bin/env python3
# File name   : tomato_config.py
# Description : Podesavanja berbe cheri paradajza (cuvaju se u tomato_config.json)
#
# Sve vrednosti koje zavise od KONKRETNOG robota (uglovi ruke/hvataljke,
# polozaj paradajza u kadru kad je izmedju prstiju hvataljke, boja...) su
# ovde, a ne rasute po kodu. Kontrolna stranica (tomato.html) ih menja
# dugmicima za kalibraciju i snima u tomato_config.json - nije potrebno
# menjati kod da bi se robot podesio.

import copy
import json
import shutil
import os
import threading

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_THIS_DIR, 'tomato_config.json')
_lock = threading.Lock()

DEFAULTS = {
    # prepisuje DETECTOR_DEFAULTS iz tomato_detector.py (boja itd.)
    'detector': {},
    'picker': {
        # --- poza "gledanja" (ruka kanal 2, zglob/kamera kanal 3), stepeni
        #     u odnosu na init poziciju servoa (isti sistem kao moveAngle) ---
        'look_arm': 0,
        'look_wrist': 0,
        # --- trazenje (sporije nego ranije - vidi snimke) ---
        'search_sweep': 60,          # +- stepeni pan servoa (kanal 1)
        'search_step': 3,
        'search_step_delay': 0.25,   # s; kamera mora da stigne da vidi novi ugao
        'search_edge_pause': 0.5,
        # --- brzine GLATKIH servo pokreta (stepeni u sekundi) ---
        # (usporeno na zahtev: spustanje ruke, hvatanje i odlaganje su sporiji)
        'pan_speed_dps': 40,
        'arm_speed_dps': 25,
        'arm_down_dps': 12,          # spustanje ruke ka paradajzu (sporo)
        'arm_lift_dps': 10,          # podizanje sa paradajzom (sporo = ne ispada)
        'wrist_speed_dps': 20,
        'claw_speed_dps': 40,        # otvaranje hvataljke pre hvatanja
        'claw_close_dps': 15,        # zatvaranje oko paradajza (sporo i do kraja)
        'claw_release_dps': 12,      # otvaranje pri odlaganju (sporo)
        'carry_speed_dps': 15,       # nosenje do mesta odlaganja
        'place_down_dps': 10,        # spustanje paradajza na pod (sporo)
        'pause_before_grab_s': 0.5,  # kratka pauza pre zatvaranja hvataljke
        'pause_after_place_s': 0.8,  # pauza posle spustanja, pre otvaranja
        'pause_after_release_s': 0.8,
        # --- kamera ---
        'hfov_deg': 60.0,            # horizontalno vidno polje USB kamere
        'pan_sign': 1,               # +1: veci ugao kanala 1 = pogled LEVO (kao 'lookleft')
        'pan_kp': 0.55,              # deo greske koji se ispravi u jednom koraku
        'pan_max_step': 4.0,
        'pan_limit': 80,
        'center_tol_frac': 0.07,     # |x - centar| < tol*sirina -> centrirano
        'settle_s': 0.12,            # posle pokreta cekaj ovoliko pre nove ocene slike
        # --- prilazak (u pozi gledanja) ---
        'approach_speed': 32,        # 0-100
        'pulse_on_s': 0.25,          # voznja u kratkim impulsima (manje zamucenja,
        'pulse_off_s': 0.2,          #  robot ne "preleti" metu)
        'min_pulse_s': 0.08,
        'steer_sign': -1,            # skretanje = steer_sign * steer_gain * pan
        'steer_gain': 0.9,           # (negativan ugao kanala 0 = levo, kao 'left')
        'steer_max': 35,
        'lost_timeout_s': 1.5,
        'max_approach_s': 45,
        'ultra_stop_cm': 7.0,
        'use_ultra': True,
        # kad je paradajz ovoliko velik (poluprecnik/sirina) ili ovoliko nisko u
        # kadru, prelazi se na spustanje ruke i fino navodjenje
        'near_r_frac': 0.045,
        'near_y_frac': 0.80,
        # --- fino navodjenje (u pozi hvatanja) ---
        'fine_speed': 26,
        'fine_pulse_s': 0.12,
        'fine_lost_s': 1.2,
        'max_fine_s': 30,
        'backup_s': 0.2,
        # --- gde je paradajz u kadru kad je TACNO izmedju prstiju hvataljke,
        #     GLEDANO IZ POZE HVATANJA. Kalibrise se JEDNIM dugmetom na stranici
        #     ("Kalibrisi hvatanje"), zajedno sa uglovima ruke/zgloba. ---
        'grab_calibrated': False,
        'grab_x_frac': 0.50,
        'grab_y_frac': 0.75,
        'grab_r_frac': 0.09,         # poluprecnik / sirina slike
        'grab_tol_frac': 0.06,
        # --- hvatanje (stepeni) ---
        # Smer ruke zavisi od montaze - zato se poza hvatanja SNIMA dugmetom
        # ("Kalibrisi hvatanje"), a podizanje je relativno (vidi lift_delta).
        # Kanal 4 'grab' (zatvori) = manji ugao.
        'claw_open': 35,
        # IZMENA: ranije -35 -> hvataljka se zatvarala samo delimicno i
        # paradajz je ispadao. Sada se zatvara (postepeno) skoro do kraja; tacna
        # vrednost za tvoju hvataljku se snima dugmetom "Zapamti zatvorenu
        # hvataljku" na stranici (posle rucnog zatvaranja sa glavne stranice).
        'claw_closed': -80,
        'claw_hold_s': 1.0,          # zadrzi stisak pre podizanja ruke
        'grab_arm': 15,              # poza hvatanja: ruka malo dole (+ = dole)
        'grab_wrist': 0,             #   (tacno se snima dugmetom "Kalibrisi hvatanje")
        'max_grab_attempts': 3,
        'grip_verify': True,         # posle podizanja proveri kamerom da li je uhvacen
        'creep_s': 0.12,             # kratak pomak napred pre zatvaranja
        # podizanje posle hvatanja: 'lift_delta' stepeni od poze hvatanja ka
        # pozi GLEDANJA, od trenutnog ugla ruke (radi bez obzira na smer montaze servoa);
        # 'lift_arm_abs' = tacan ugao ako se zeli (None = koristi lift_delta)
        'lift_delta': 35,
        'lift_arm_abs': None,
        'lift_arm': -30,             # (koristi se samo ako je poza hvatanja ~0)
        # --- odlaganje ---
        # 'bucket' = robot nadje zelenu kanticu, dodje do nje i ubaci paradajz;
        # 'pose'   = paradajz se polako ostavi na snimljenu POZU ODLAGANJA.
        # Ako kantica nije kalibrisana ili nije pronadjena, robot paradajz
        # ostavi na pozu odlaganja (nikad ne ostaje zaglavljen sa njim).
        'deposit_mode': 'bucket',
        # poza odlaganja (snima se dugmetom): okret ruke, ruka, zglob
        'place_calibrated': False,
        'place_pan': None,           # None = basket_pan
        'place_arm': None,           # None = poza hvatanja (uz pod)
        'place_wrist': None,
        'basket_pan': 70,            # okretanje ruke ka mestu odlaganja na pod (kanal 1)
        'basket_arm': None,          # None = paradajz se POLAKO spusta na pod (poza hvatanja)
        'basket_wrist': 0,
        'servo_settle_s': 0.6,
        # --- kantica (kalibrise se dugmetom "KALIBRISI KANTICU": robot stoji
        #     ispred kantice na rastojanju na kom treba da ubaci paradajz) ---
        'bucket_calibrated': False,
        'bucket_near_w_frac': 0.45,  # sirina kantice / sirina kadra kad je robot dovoljno blizu
        'bucket_x_frac': 0.50,       # gde je sredina kantice u kadru kad je hvataljka iznad nje
        'bucket_ultra_cm': 12.0,     # ultrazvuk: blize od ovoga = stigao do kantice
        'carry_arm': None,           # poza nosenja/trazenja kantice (None = poza gledanja)
        'carry_wrist': None,
        'drop_arm': None,            # poza ubacivanja - ruka iznad otvora kantice
        'drop_wrist': None,          #   (None = podignuta ruka, vidi lift_delta)
        'bucket_approach_speed': 30,
        'bucket_search_timeout_s': 60,
        'bucket_max_approach_s': 45,
        'bucket_lost_s': 2.0,
        'bucket_arrive_frac': 0.75,  # kantica nestala izbliza (>= ovoliko od ciljne sirine) = stigao
        'bucket_turn_s': 0.7,        # posle punog okreta pogleda bez kantice: kratka voznja u luku
        'drop_backup_s': 0.6,        # posle ubacivanja malo unazad (da se ruka ne zakaci)
        # --- snimanje podataka za dalje treniranje modela ---
        'log_enabled': False,
        'log_interval_s': 2.0,
        'log_max_frames': 800,
    },
    # prepisuje BUCKET_DEFAULTS iz bucket_detector.py (boja kantice)
    'bucket': {},
    # --- prepoznavanje vlasnika i pozdrav ("mahanje") ---
    'face': {
        'recognize': False,          # prepoznavanje osoba (ime na videu i stranici)
        'wave_enabled': False,       # mahanje prepoznatim osobama
        'threshold': 0.40,           # kosinusna slicnost SFace otiska (0.363 = preporuka OpenCV-a)
        'margin': 0.06,              # najbolja osoba mora biti bar ovoliko ispred druge
        'greet_cooldown_s': 30,      # ista osoba se ponovo "pozdravlja" posle ovoliko sekundi
        'confirm_hits': 2,
        'min_face_px': 36,
        'interval_s': 0.35,          # analiza lica ~3x u sekundi (samo kad je potrebno)
        'wave_cooldown_s': 20,       # najmanji razmak izmedju dva pozdrava
        'wave_cycles': 3,            # koliko puta levo-desno
        'wave_amplitude': 25,        # stepeni pan servoa (kanal 1) levo i desno
        'wave_speed_dps': 90,
        'wave_arm_lift': 20,         # ruka se malo podigne pre mahanja (stepeni)
        'wave_claw': True,           # hvataljka se otvara/zatvara uz mahanje ("dlan")
        'turn_to_face': True,        # pre mahanja okreni "glavu" ka vlasniku
        'watch_arm': None,           # poza u kojoj robot ceka lica (None = trenutna)
        'watch_wrist': None,
    },
    # --- kontrolna stranica ---
    'ui': {
        'control_mode': 'auto',      # 'auto' (samostalna berba) ili 'manual' (rucni koraci)
        'assistant': True,           # AI asistent (preporuke) ukljucen
        'voice': False,              # glasovne poruke na stranici
    },
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise ValueError('nije JSON objekat')
    return d


def load():
    with _lock:
        user = {}
        if os.path.exists(CONFIG_PATH):
            try:
                user = _read_json(CONFIG_PATH)
            except Exception as e:
                # pokvaren fajl (npr. nestanak struje) - NE gubi kalibraciju:
                # vrati poslednju ispravnu kopiju (.bak)
                print('[TOMATO] tomato_config.json je neispravan (%s) - koristim rezervnu kopiju.' % e)
                try:
                    user = _read_json(CONFIG_PATH + '.bak')
                except Exception:
                    user = {}
    cfg = _merge(DEFAULTS, user)
    # prethodna verzija je imala 'side' (odlaganje na pod) - sada je to 'pose'
    if cfg['picker'].get('deposit_mode') not in ('bucket', 'pose'):
        cfg['picker']['deposit_mode'] = 'pose'
    if cfg['ui'].get('control_mode') not in ('auto', 'manual'):
        cfg['ui']['control_mode'] = 'auto'
    return cfg


def _diff(cfg, base):
    out = {}
    for k, v in cfg.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            out[k] = _diff(v, base[k])
        elif k not in base or base[k] != v:
            out[k] = v
    return out


def save(cfg):
    """Snima SAMO vrednosti koje se razlikuju od podrazumevanih (DEFAULTS),
    tako da buduce ispravke podrazumevanih vrednosti u kodu stignu do
    robota. Atomicno snimanje (prvo u privremeni fajl) - nestanak struje u
    toku snimanja ne moze da ostavi pokvaren JSON."""
    with _lock:
        tmp = CONFIG_PATH + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(_diff(cfg, DEFAULTS), f, indent=2, sort_keys=True, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        # rezervna kopija poslednjeg ISPRAVNOG fajla
        try:
            _read_json(CONFIG_PATH)
            shutil.copyfile(CONFIG_PATH, CONFIG_PATH + '.bak')
        except Exception:
            pass
        os.replace(tmp, CONFIG_PATH)


# ----------------------------------------------------------------------
# Podesavanja koja se menjaju sa kontrolne stranice (tab "Podesavanja").
# (sekcija, kljuc, naziv, grupa, tip, min, max, korak, jedinica)
# Server prihvata SAMO ove kljuceve i SAMO u zadatim granicama.
# ----------------------------------------------------------------------
SETTINGS = [
    ('picker', 'pan_speed_dps', 'Okretanje ruke (pan)', 'Brzine pokreta', 'int', 5, 120, 1, '°/s'),
    ('picker', 'arm_speed_dps', 'Ruka - opšte', 'Brzine pokreta', 'int', 5, 90, 1, '°/s'),
    ('picker', 'arm_down_dps', 'Spuštanje ruke ka paradajzu', 'Brzine pokreta', 'int', 3, 60, 1, '°/s'),
    ('picker', 'arm_lift_dps', 'Podizanje sa paradajzom', 'Brzine pokreta', 'int', 3, 60, 1, '°/s'),
    ('picker', 'wrist_speed_dps', 'Zglob', 'Brzine pokreta', 'int', 5, 90, 1, '°/s'),
    ('picker', 'claw_close_dps', 'Zatvaranje hvataljke', 'Brzine pokreta', 'int', 3, 90, 1, '°/s'),
    ('picker', 'claw_release_dps', 'Otvaranje pri odlaganju', 'Brzine pokreta', 'int', 3, 90, 1, '°/s'),
    ('picker', 'carry_speed_dps', 'Nošenje paradajza', 'Brzine pokreta', 'int', 3, 90, 1, '°/s'),
    ('picker', 'place_down_dps', 'Spuštanje na pod', 'Brzine pokreta', 'int', 3, 60, 1, '°/s'),
    ('picker', 'search_sweep', 'Ugao traženja (±)', 'Traženje i prilazak', 'int', 10, 80, 1, '°'),
    ('picker', 'search_step', 'Korak traženja', 'Traženje i prilazak', 'int', 1, 10, 1, '°'),
    ('picker', 'approach_speed', 'Brzina prilaska', 'Traženje i prilazak', 'int', 15, 70, 1, '%'),
    ('picker', 'fine_speed', 'Brzina finog navođenja', 'Traženje i prilazak', 'int', 15, 60, 1, '%'),
    ('picker', 'max_grab_attempts', 'Pokušaja hvatanja', 'Traženje i prilazak', 'int', 1, 6, 1, ''),
    ('picker', 'grip_verify', 'Provera hvatanja kamerom', 'Traženje i prilazak', 'bool', None, None, None, ''),
    ('picker', 'use_ultra', 'Ultrazvučna kočnica', 'Traženje i prilazak', 'bool', None, None, None, ''),
    ('picker', 'ultra_stop_cm', 'Ultrazvuk - stop', 'Traženje i prilazak', 'float', 3, 40, 0.5, 'cm'),
    ('picker', 'pause_before_grab_s', 'Pauza pre hvatanja', 'Pauze', 'float', 0, 3, 0.1, 's'),
    ('picker', 'claw_hold_s', 'Držanje stiska', 'Pauze', 'float', 0, 3, 0.1, 's'),
    ('picker', 'pause_after_place_s', 'Pauza pre otvaranja', 'Pauze', 'float', 0, 3, 0.1, 's'),
    ('picker', 'pause_after_release_s', 'Pauza posle otvaranja', 'Pauze', 'float', 0, 3, 0.1, 's'),
    ('picker', 'deposit_mode', 'Odlaganje paradajza', 'Kantica', 'choice', ['bucket', 'pose'], None, None, ''),
    ('picker', 'bucket_approach_speed', 'Brzina prilaska kantici', 'Kantica', 'int', 15, 70, 1, '%'),
    ('picker', 'bucket_search_timeout_s', 'Najduže traženje kantice', 'Kantica', 'int', 10, 300, 5, 's'),
    ('picker', 'bucket_ultra_cm', 'Ultrazvuk - stani ispred kantice', 'Kantica', 'float', 4, 40, 0.5, 'cm'),
    ('picker', 'drop_backup_s', 'Unazad posle ubacivanja', 'Kantica', 'float', 0, 2, 0.1, 's'),
    ('picker', 'bucket_arrive_frac', 'Dolazak kad kantica izađe iz kadra (od cilja)', 'Kantica', 'float', 0.4, 1.0, 0.05, ''),
    ('face', 'threshold', 'Prag prepoznavanja lica', 'Osobe i pozdrav', 'float', 0.30, 0.70, 0.01, ''),
    ('face', 'margin', 'Razlika između dve osobe', 'Osobe i pozdrav', 'float', 0.0, 0.20, 0.01, ''),
    ('face', 'greet_cooldown_s', 'Ponovo javi istu osobu posle', 'Osobe i pozdrav', 'int', 5, 600, 5, 's'),
    ('face', 'wave_cooldown_s', 'Razmak između pozdrava', 'Osobe i pozdrav', 'int', 5, 300, 5, 's'),
    ('face', 'wave_cycles', 'Broj mahanja', 'Osobe i pozdrav', 'int', 1, 6, 1, ''),
    ('face', 'wave_amplitude', 'Širina mahanja (±)', 'Osobe i pozdrav', 'int', 10, 45, 1, '°'),
    ('face', 'wave_speed_dps', 'Brzina mahanja', 'Osobe i pozdrav', 'int', 30, 180, 5, '°/s'),
    ('face', 'wave_arm_lift', 'Podizanje ruke pri mahanju', 'Osobe i pozdrav', 'int', 0, 45, 1, '°'),
    ('face', 'wave_claw', 'Hvataljka "maše" (otvori/zatvori)', 'Osobe i pozdrav', 'bool', None, None, None, ''),
    ('face', 'turn_to_face', 'Okreni se ka vlasniku', 'Osobe i pozdrav', 'bool', None, None, None, ''),
]
_SETTINGS_INDEX = {(s[0], s[1]): s for s in SETTINGS}


def settings_schema(cfg=None):
    """Lista podesavanja sa trenutnim vrednostima (za stranicu)."""
    cfg = cfg or load()
    out = []
    for sec, key, label, group, typ, lo, hi, step, unit in SETTINGS:
        item = {'id': '%s.%s' % (sec, key), 'label': label, 'group': group, 'type': typ,
                'unit': unit, 'value': cfg.get(sec, {}).get(key),
                'default': DEFAULTS[sec].get(key)}
        if typ == 'choice':
            item['options'] = lo
        else:
            item.update({'min': lo, 'max': hi, 'step': step})
        out.append(item)
    return out


def validate_setting(sid, value):
    """Proverava jednu vrednost sa stranice. Vraca (sekcija, kljuc, vrednost)
    ili baca ValueError sa porukom za korisnika."""
    try:
        sec, key = str(sid).split('.', 1)
    except ValueError:
        raise ValueError('Nepoznato podešavanje: %s' % sid)
    spec = _SETTINGS_INDEX.get((sec, key))
    if spec is None:
        raise ValueError('Nepoznato podešavanje: %s' % sid)
    _, _, label, _, typ, lo, hi, _, _ = spec
    if typ == 'bool':
        if isinstance(value, bool):
            return sec, key, value
        if str(value).lower() in ('1', 'true', 'on', 'da'):
            return sec, key, True
        if str(value).lower() in ('0', 'false', 'off', 'ne'):
            return sec, key, False
        raise ValueError('%s: ocekuje se da/ne' % label)
    if typ == 'choice':
        if value not in lo:
            raise ValueError('%s: dozvoljeno je %s' % (label, ', '.join(lo)))
        return sec, key, value
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError('%s: nije broj' % label)
    if v != v or v < lo or v > hi:
        raise ValueError('%s: dozvoljeno %s - %s' % (label, lo, hi))
    return sec, key, (int(round(v)) if typ == 'int' else round(v, 3))


def apply_settings(changes):
    """changes: {'picker.arm_down_dps': 12, ...}. Sve ili nista: ako je i
    jedna vrednost neispravna nista se ne snima. Vraca novu konfiguraciju."""
    if not isinstance(changes, dict) or not changes:
        raise ValueError('Nema promena.')
    valid = [validate_setting(k, v) for k, v in changes.items()]
    cfg = load()
    for sec, key, v in valid:
        cfg.setdefault(sec, {})[key] = v
    save(cfg)
    return cfg


def reset_settings():
    """Vraca sva podesavanja sa taba "Podesavanja" na podrazumevana
    (kalibracije - poze, boje, hvataljka - se NE diraju)."""
    cfg = load()
    for sec, key, *_ in SETTINGS:
        cfg[sec][key] = copy.deepcopy(DEFAULTS[sec][key])
    save(cfg)
    return cfg
