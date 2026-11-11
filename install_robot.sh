#!/bin/bash
# =====================================================================
#  install_robot.sh - vraca KOMPLETAN softver robota PiCar-Pro na novi
#  Raspberry Pi OS (posle ostecenja SD kartice).
#
#  Pokretanje (u PuTTY-ju, iz foldera robota):
#      cd ~/Adeept_PiCar-Pro-main
#      sudo bash install_robot.sh
#
#  Sta radi:
#   1. proverava da su svi fajlovi robota tu i ispravni
#   2. ukljucuje I2C (servo ploca, ekran) i SPI (LED trake)
#   3. instalira sve potrebne pakete (kao Adeept setup.py)
#   4. vraca naucena lica (rebuild_faces.py)
#   5. podesava automatsko pokretanje robota i Adeept_Robot hotspot
#   6. proverava da se sve ucitava, pa restartuje Raspberry Pi
#  Ceo zapis rada: /var/log/robot_install.log
# =====================================================================

set -u
if [ "$(id -u)" -ne 0 ]; then
    echo "Pokreni sa sudo:   sudo bash install_robot.sh"
    exit 1
fi

ROBOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER="$ROBOT_DIR/Server"
USER_NAME="${SUDO_USER:-ado}"
USER_HOME="$(getent passwd "$USER_NAME" | cut -d: -f6)"
LOG=/var/log/robot_install.log
exec > >(tee -a "$LOG") 2>&1

ok()   { echo -e "  \e[32mOK\e[0m  $*"; }
warn() { echo -e "  \e[33mPAZNJA\e[0m  $*"; WARNINGS=$((WARNINGS+1)); }
fail() { echo -e "\n  \e[31mGRESKA\e[0m  $*\n"; exit 1; }
WARNINGS=0

echo "================================================================"
echo " Vracanje softvera robota  ($(date '+%d.%m.%Y %H:%M'))"
echo " folder: $ROBOT_DIR    korisnik: $USER_NAME"
echo "================================================================"

# ---------------------------------------------------------------- 1
echo; echo "[1/6] Provera fajlova robota"
[ -f "$SERVER/WebServer.py" ] || fail "Nema Server/WebServer.py - pokreni skriptu iz foldera robota."
REQUIRED="WebServer.py app.py TomatoPicker.py tomato_detector.py tomato_classifier.py
tomato_config.py tomato_datalog.py bucket_detector.py face_module.py assistant.py
camera_opencv.py base_camera.py RPIservo.py Move.py OLED.py Functions.py Switch.py
RobotLight.py Voltage.py Info.py Ultra.py Buzzer.py tomato_model.json
tomato_clf_weights.json dist/tomato.html dist/index.html
models/face_detection_yunet_2023mar.onnx models/face_recognition_sface_2021dec.onnx"
for f in $REQUIRED; do
    [ -s "$SERVER/$f" ] || fail "Nedostaje ili je prazan fajl Server/$f"
done
ok "svi fajlovi su na mestu"

SFACE_SUM=0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79
if [ "$(sha256sum "$SERVER/models/face_recognition_sface_2021dec.onnx" | cut -d' ' -f1)" = "$SFACE_SUM" ]; then
    ok "model za prepoznavanje lica je ispravan"
else
    warn "model SFace je ostecen - prepoznavanje lica nece raditi (posalji ga ponovo preko WinSCP-a)"
fi

# ciscenje ostataka sa stare kartice i Windows prelaza redova
find "$ROBOT_DIR" -name "__pycache__" -type d -prune -exec rm -rf {} +
for f in "$ROBOT_DIR"/*.sh; do sed -i 's/\r$//' "$f"; done
chmod +x "$ROBOT_DIR"/*.sh
chown -R "$USER_NAME":"$USER_NAME" "$ROBOT_DIR"
ok "ociscen kes i podesena prava"

# ---------------------------------------------------------------- 2
echo; echo "[2/6] Ukljucivanje I2C i SPI"
raspi-config nonint do_i2c 0 && ok "I2C ukljucen" || warn "I2C nije ukljucen (raspi-config)"
raspi-config nonint do_spi 0 && ok "SPI ukljucen" || warn "SPI nije ukljucen (raspi-config)"

# ---------------------------------------------------------------- 3
echo; echo "[3/6] Instalacija paketa (traje 5-15 minuta, ne gasi robota)"
export DEBIAN_FRONTEND=noninteractive
for try in 1 2 3; do
    apt-get update && apt-get install -y \
        i2c-tools python3-smbus python3-gpiozero python3-picamera2 \
        python3-opencv python3-numpy python3-pip && APT_OK=1 && break
    APT_OK=0; echo "  ponovni pokusaj ($try)..."; sleep 5
done
[ "${APT_OK:-0}" = 1 ] && ok "sistemski paketi" || fail "apt instalacija nije uspela - proveri internet pa pokreni ponovo"
# dodatni paketi iz Adeept uputstva - pojedinacno, da jedan koji fali ne zaustavi sve
for p in opencv-data python3-pigpio python3-pyaudio python3-psutil v4l-utils; do
    apt-get install -y "$p" >/dev/null 2>&1 && ok "$p" || echo "  ($p nije dostupan - nije neophodan)"
done

# websockets 13.1: WebServer.py koristi handler (websocket, path) i
# get_event_loop().run_until_complete(serve(...)), sto radi samo do v13
PIP_PKGS="adafruit-circuitpython-motor adafruit-circuitpython-pca9685 luma.oled
mpu6050-raspberrypi spidev adafruit-circuitpython-busdevice adafruit-circuitpython-ssd1306
pillow flask flask_cors imutils pyzmq pybase64 psutil websockets==13.1 adafruit-circuitpython-ads7830"
for try in 1 2 3; do
    pip3 install --break-system-packages $PIP_PKGS && PIP_OK=1 && break
    PIP_OK=0; echo "  ponovni pokusaj ($try)..."; sleep 5
done
[ "${PIP_OK:-0}" = 1 ] && ok "Python paketi" || fail "pip instalacija nije uspela - proveri internet pa pokreni ponovo"
# glasovne komande iz Adeept primera - nisu potrebne za berbu, zato nisu obavezne
pip3 install --break-system-packages sherpa-onnx sherpa-onnx-bin >/dev/null 2>&1 \
    && ok "sherpa-onnx (glas)" || warn "sherpa-onnx (glasovni primeri) nije instaliran - berba radi i bez njega"

# ---------------------------------------------------------------- 4
echo; echo "[4/6] Vracanje naucenih lica"
( cd "$SERVER" && python3 rebuild_faces.py ) || warn "lica nisu vracena - nauci ih ponovo na stranici"
chown -R "$USER_NAME":"$USER_NAME" "$SERVER/faces" 2>/dev/null
if [ -s "$SERVER/tomato_config.json" ]; then
    ok "kalibracija (tacka hvatanja, kanta, poze) vracena iz tomato_config.json"
else
    warn "nema tomato_config.json - kalibraciju uradi ponovo na stranici"
fi

# ---------------------------------------------------------------- 5
echo; echo "[5/6] Automatsko pokretanje"
cp "$ROBOT_DIR/wifi_hotspot_manager.sh" "$USER_HOME/wifi_hotspot_manager.sh"
chmod +x "$USER_HOME/wifi_hotspot_manager.sh"
cat > /etc/systemd/system/wifi-hotspot-manager.service <<EOF
[Unit]
Description=WiFi and Hotspot Manager Service
After=network.target NetworkManager.service
Wants=NetworkManager.service

[Service]
Type=oneshot
ExecStart=$USER_HOME/wifi_hotspot_manager.sh
User=root
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

cat > "$USER_HOME/startup.sh" <<EOF
#!/bin/sh
sleep 5
cd $SERVER
exec python3 $SERVER/WebServer.py
EOF
chmod 755 "$USER_HOME/startup.sh"
chown "$USER_NAME":"$USER_NAME" "$USER_HOME/startup.sh"

cat > /etc/systemd/system/Adeept_Robot.service <<EOF
[Unit]
Description=Auto-start robot control script
After=wifi-hotspot-manager.service

[Service]
Type=simple
User=root
WorkingDirectory=$SERVER
ExecStart=$USER_HOME/startup.sh
Restart=no

[Install]
WantedBy=multi-user.target
EOF
chmod 644 /etc/systemd/system/wifi-hotspot-manager.service /etc/systemd/system/Adeept_Robot.service
systemctl daemon-reload
systemctl enable wifi-hotspot-manager.service Adeept_Robot.service >/dev/null 2>&1 \
    && ok "Adeept_Robot.service i wifi-hotspot-manager.service ukljuceni" \
    || warn "servisi nisu ukljuceni"

# ---------------------------------------------------------------- 6
echo; echo "[6/6] Zavrsna provera"
python3 - <<'EOF' || WARNINGS=$((WARNINGS+1))
import importlib, sys
bad = []
for m in ['cv2', 'numpy', 'flask', 'flask_cors', 'websockets', 'imutils', 'psutil', 'zmq',
          'luma.oled.device', 'adafruit_pca9685', 'adafruit_motor.servo', 'board', 'busio', 'spidev']:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append('%s (%s)' % (m, e))
try:
    import cv2, websockets
    print('  OpenCV %s, websockets %s, YuNet: %s' % (cv2.__version__, websockets.__version__,
          'da' if hasattr(cv2, 'FaceDetectorYN') else 'NE'))
    if not websockets.__version__.startswith('13.'):
        bad.append('websockets mora biti 13.x (ima %s)' % websockets.__version__)
except Exception:
    pass
if bad:
    print('  NE UCITAVA SE: ' + ', '.join(bad)); sys.exit(1)
print('  svi paketi se ucitavaju')
EOF
( cd "$SERVER" && python3 -m py_compile *.py ) && ok "sav kod robota se prevodi bez greske" \
    || warn "neki fajl koda ima gresku (vidi iznad)"
ls /dev/video* >/dev/null 2>&1 && ok "USB kamera je prikljucena" || warn "kamera nije pronadjena (/dev/video*) - proveri USB kabl"
i2cdetect -y 1 2>/dev/null | grep -q " 40 " && ok "servo ploca (PCA9685) vidljiva na I2C" \
    || echo "  (servo ploca jos nije vidljiva - normalno pre restarta posle ukljucivanja I2C)"

echo
echo "================================================================"
if [ "$WARNINGS" -eq 0 ]; then
    echo " GOTOVO bez upozorenja."
else
    echo " GOTOVO, uz $WARNINGS upozorenja (vidi iznad ili $LOG)."
fi
echo " Posle restarta otvori:  http://Adnan.local:5000/tomato.html"
echo " Lica: kartica 'Osobe' -> upisi prava imena umesto 'Osoba 1, 2...'"
echo " UVEK gasi robota sa:    sudo shutdown now   (pa tek onda prekidac)"
echo "================================================================"
echo "Restart za 15 sekundi (Ctrl+C da otkazes)..."
sleep 15
reboot
