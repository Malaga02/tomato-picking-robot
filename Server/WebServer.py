#!/usr/bin/env/python
# File name   : WebServer.py
# Website     : www.Adeept.com
# Author      : Adeept
# Date        : 2025/03/11
import time
import threading
import Move as move
import os
import Info as info
import RPIservo
import Functions as functions
import RobotLight as robotLight
import Switch as switch
import socket
import asyncio
import websockets
# IZMENA: modul za glasovne komande (Voice_Command) nedostaje/nepotpun je u
# ovoj verziji zvanicnog Adeept softvera (u Server/ postoji samo
# VoiceIdentify.py sa klasom Speech, bez klase Sherpa_ncnn koju kod ispod
# ocekuje) - bez ovog izmene ceo WebServer.py bi se rusio vec pri pokretanju.
# Glasovne komande nisu deo ovog master rada (beranje paradajza), zato uvoz
# obavijamo u try/except da ostatak servisa (kamera, motori, TomatoPicker)
# normalno radi i kada Voice_Command nije dostupan.
try:
    import Voice_Command
    VOICE_COMMAND_OK = True
except ImportError:
    Voice_Command = None
    VOICE_COMMAND_OK = False
import json
import app
import Voltage
import subprocess
import TomatoPicker


Dv = -1 #Directional variable
OLED_connection = 1

try:
    import OLED
    screen = OLED.OLED_ctrl()
    screen.start()
    screen.screen_show(1, 'ADEEPT.COM')
except:
    OLED_connection = 0
    print('OLED disconnected')
    pass


functionMode = 0
speed_set = 50
rad = 0.5


scGear = RPIservo.ServoCtrl()
scGear.moveInit()
scGear.start()

modeSelect = 'PT'

init_pwm0 = scGear.initPos[0]
init_pwm1 = scGear.initPos[1]
init_pwm2 = scGear.initPos[2]
init_pwm3 = scGear.initPos[3]
init_pwm4 = scGear.initPos[4]

fuc = functions.Functions()
fuc.setup()
fuc.start()

# IZMENA: autonomni rezim za prepoznavanje i branje cheri paradajza (master
# rad) - vidi TomatoPicker.py za ceo opis logike i podesavanja kalibracije.
tomatoPicker = TomatoPicker.TomatoPicker(scGear)
tomatoPicker.start()

# IZMENA: pokreni glasovne komande samo ako je modul uspesno ucitan (vidi
# napomenu kod uvoza na vrhu fajla) - u suprotnom SR ostaje None.
SR = None
if VOICE_COMMAND_OK:
    try:
        ncnn = Voice_Command.Sherpa_ncnn()
        ncnn.start()
        SR = Voice_Command.Speech()
        SR.start()
    except Exception as e:
        print('Voice_Command se ne pokreće (preskočeno):', e)
        SR = None

batteryMonitor = Voltage.BatteryLevelMonitor()
batteryMonitor.start()

curpath = os.path.realpath(__file__)
thisPath = "/" + os.path.dirname(curpath)


def servoPosInit():
    scGear.initConfig(0,init_pwm0,1)
    scGear.initConfig(1,init_pwm1,1)
    scGear.initConfig(2,init_pwm2,1)
    scGear.initConfig(3,init_pwm3,1)
    scGear.initConfig(4,init_pwm4,1)


def replace_num(initial,new_num):   #Call this function to replace data in '.txt' file
    global r
    newline=""
    str_num=str(new_num)
    with open(thisPath+"/RPIservo.py","r") as f:
        for line in f.readlines():
            if(line.find(initial) == 0):
                line = initial+"%s" %(str_num+"\n")
            newline += line
    with open(thisPath+"/RPIservo.py","w") as f:
        f.writelines(newline)


def functionSelect(command_input, response):
    if 'scan' == command_input:
        if OLED_connection:
            screen.screen_show(5,'SCANNING')
        if modeSelect == 'PT':
            radar_send = fuc.radarScan()
            response['title'] = 'scanResult'
            response['data'] = radar_send
            time.sleep(0.3)

    elif 'findColor' == command_input:
        if OLED_connection:
            screen.screen_show(5,'FindColor')
        if modeSelect == 'PT':
            flask_app.modeselect('findColor')

    elif 'motionGet' == command_input:
        if OLED_connection:
            screen.screen_show(5,'MotionGet')
        flask_app.modeselect('watchDog')

    elif 'stopCV' == command_input:
        if OLED_connection:
            screen.screen_show(5,'FUNCTION OFF')
        flask_app.modeselect('none')
        switch.switch(1,0)
        switch.switch(2,0)
        switch.switch(3,0)
        move.motorStop()

    elif 'automaticOff' == command_input:
        if OLED_connection:
            screen.screen_show(5,'FUNCTION OFF')
        ws2812.pause()
        fuc.pause()
        move.motorStop()
        time.sleep(0.3)
        move.motorStop()

    elif 'automatic' == command_input:
        if OLED_connection:
            screen.screen_show(5,'Automatic')
        if modeSelect == 'PT':
            fuc.automatic()
        else:
            fuc.pause()

    elif 'trackLine' == command_input:
        functions.last_status = None
        fuc.trackLine()
        if OLED_connection:
            screen.screen_show(5,'TrackLine')

    elif 'trackLineOff' == command_input:
        if OLED_connection:
            screen.screen_show(5,'FUNCTION OFF')
        fuc.pause()
        move.motorStop()

    elif 'steadyCamera' == command_input:
        if OLED_connection:
            screen.screen_show(5,'SteadyCamera')
        fuc.steady(scGear.lastPos[2])

    elif 'steadyCameraOff' == command_input:
        if OLED_connection:
            screen.screen_show(5,'FUNCTION OFF')
        fuc.pause()
        move.motorStop()

    elif 'police' == command_input:
        if OLED_connection:
            screen.screen_show(5,'POLICE')
        ws2812.police()
        pass

    elif 'policeOff' == command_input:
        if OLED_connection:
            screen.screen_show(5,'FUNCTION OFF')
        ws2812.breath(75,85,90)
        pass
    
    elif 'speech' == command_input:
        if OLED_connection:
            screen.screen_show(5,'SPEECH')
        if SR is not None:  # IZMENA: SR moze biti None ako Voice_Command nije dostupan
            SR.speech()
        pass

    elif 'speechOff' == command_input:
        if OLED_connection:
            screen.screen_show(5,'FUNCTION OFF')
        if SR is not None:  # IZMENA: SR moze biti None ako Voice_Command nije dostupan
            SR.pause()
        pass
    
    elif 'keepDistance' == command_input:
        functions.last_status = 25
        fuc.keepDistance()
        if OLED_connection:
            screen.screen_show(5,'KeepDistance')

    elif 'keepDistanceOff' == command_input:
        if OLED_connection:
            screen.screen_show(5,'FUNCTION OFF')
        fuc.pause()
        move.motorStop()
        time.sleep(0.5)
        move.motorStop()

    # IZMENA: komande za autonomno branje cheri paradajza (master rad) -
    # vidi TomatoPicker.py
    elif 'tomatoStart' == command_input:
        if OLED_connection:
            screen.screen_show(5,'TOMATO PICK')
        msg = tomatoPicker.startPicking()
        if msg and msg != 'Berba pokrenuta.':
            response['title'] = 'tomatoMsg'
            response['data'] = msg

    elif 'tomatoStop' == command_input:
        if OLED_connection:
            screen.screen_show(5,'FUNCTION OFF')
        tomatoPicker.stopPicking()

def switchCtrl(command_input, response):
    if 'Switch_1_on' in command_input:
        switch.switch(1,1)

    elif 'Switch_1_off' in command_input:
        switch.switch(1,0)

    elif 'Switch_2_on' in command_input:
        switch.switch(2,1)

    elif 'Switch_2_off' in command_input:
        switch.switch(2,0)

    elif 'Switch_3_on' in command_input:
        switch.switch(3,1)

    elif 'Switch_3_off' in command_input:
        switch.switch(3,0) 


def robotCtrl(command_input, response):
    if 'forward' == command_input:
        move.move(speed_set, 1, "mid")
    
    elif 'backward' == command_input:
        move.move(speed_set, -1, "mid")

    elif 'DS' in command_input:
        move.motorStop()

    elif 'left' == command_input:
        scGear.moveAngle(0, 30  * Dv)
        time.sleep(0.15)
        move.move(30, 1, "mid")
        switch.switch(3,1)
        time.sleep(0.15)

    elif 'right' == command_input:
        scGear.moveAngle(0,-30  * Dv)
        time.sleep(0.15)
        move.move(30, 1, "mid")
        switch.switch(2,1)
        time.sleep(0.15)
    elif 'TS' in command_input:
        scGear.moveAngle(0, 0)
        move.motorStop()
        switch.switch(2,0)
        switch.switch(3,0)

    elif 'lookleft' == command_input:
        scGear.singleServo(1, 1, 7)

    elif 'lookright' == command_input:
        scGear.singleServo(1,-1, 7)

    elif 'LRstop' in command_input:
        scGear.stopWiggle()


    elif 'armup' == command_input:
        scGear.singleServo(2, -1, 7)

    elif 'armdown' == command_input:
        scGear.singleServo(2, 1, 7)

    elif 'armstop' in command_input:
        scGear.stopWiggle()



    elif 'handup' == command_input:
        scGear.singleServo(3, 1, 7)

    elif 'handdown' == command_input:
        scGear.singleServo(3, -1, 7)

    elif 'HAstop' in command_input:
        scGear.stopWiggle()

    elif 'grab' == command_input:
        scGear.singleServo(4, -1, 7)

    elif 'loose' == command_input:
        scGear.singleServo(4, 1, 7)

    elif 'stop' == command_input:
        scGear.stopWiggle()
    elif 'home' == command_input:
        scGear.moveServoInit([0])
        scGear.moveServoInit([1])
        scGear.moveServoInit([2])
        scGear.moveServoInit([3])

def configPWM(command_input, response):
    global init_pwm0, init_pwm1, init_pwm2, init_pwm3, init_pwm4

    if 'SiLeft' in command_input:
        numServo = int(command_input[7:])
        if numServo == 0:
            init_pwm0 -= 2
            scGear.setPWM(0,init_pwm0)
        elif numServo == 1:
            init_pwm1 -= 2
            scGear.setPWM(1,init_pwm1)
        elif numServo == 2:
            init_pwm2 -= 2
            scGear.setPWM(2,init_pwm2)
        elif numServo == 3:
            init_pwm3 -= 2
            scGear.setPWM(3,init_pwm3)
        elif numServo == 4:
            init_pwm4 -= 2
            scGear.setPWM(4,init_pwm4)

    if 'SiRight' in command_input:
        numServo = int(command_input[8:])
        if numServo == 0:
            init_pwm0 += 2
            scGear.setPWM(0,init_pwm0)
        elif numServo == 1:
            init_pwm1 += 2
            scGear.setPWM(1,init_pwm1)
        elif numServo == 2:
            init_pwm2 += 2
            scGear.setPWM(2,init_pwm2)
        elif numServo == 3:
            init_pwm3 += 2
            scGear.setPWM(3,init_pwm3)
        elif numServo == 4:
            init_pwm4 += 2
            scGear.setPWM(4,init_pwm4)

    if 'PWMMS' in command_input:
        numServo = int(command_input[6:])
        scGear.moveAngle(numServo, 0)


    if 'PWMINIT' == command_input:
        print(init_pwm1)
        servoPosInit()
    elif 'PWMD' in command_input:
        init_pwm0 = 90 
        init_pwm1 = 90 
        init_pwm2 = 90 
        init_pwm3 = 90 
        init_pwm4 = 90
        for i in range(5):
            scGear.moveAngle(i, 0)

def tomatoCommand(cmd):
    # IZMENA: komande sa tomato.html (vidi TomatoPicker.py / tomato_config.py)
    tp = tomatoPicker
    simple = {
        # kalibracija
        'tomatoSaveGrabTarget': tp.saveGrabTarget,
        'tomatoSaveGrabPose': tp.saveGrabPose,
        'tomatoSaveLookPose': tp.saveLookPose,
        'tomatoSaveBucket': tp.saveBucketTarget,
        'tomatoSaveDropPose': tp.saveDropPose,
        'tomatoSaveWatchPose': tp.saveWatchPose,
        'tomatoColorReset': tp.resetColor,
        'tomatoBucketColorReset': tp.resetBucketColor,
        'tomatoClawTest': tp.testClaw,
        'tomatoSaveClawClosed': lambda: tp.saveClaw('closed'),
        'tomatoSaveClawOpen': lambda: tp.saveClaw('open'),
        'tomatoSavePlacePose': tp.savePlacePose,
        'tomatoCalibClawMask': tp.calibrateClawMask,
        'tomatoResetClawMask': tp.resetClawMask,
        'tomatoGoLook': lambda: tp.goToPose('look'),
        'tomatoGoGrab': lambda: tp.goToPose('grab'),
        'tomatoGoDrop': lambda: tp.goToPose('drop'),
        'tomatoGoPlace': lambda: tp.goToPose('place'),
        'tomatoGoWatch': lambda: tp.goToPose('watch'),
        # osobe i pozdrav
        'tomatoRecogOn': lambda: tp.setRecognize(True),
        'tomatoRecogOff': lambda: tp.setRecognize(False),
        'tomatoWaveOn': lambda: tp.setWave(True),
        'tomatoWaveOff': lambda: tp.setWave(False),
        'tomatoWaveTest': tp.testWave,
        'tomatoFaceCancel': tp.cancelEnroll,
        'tomatoFaceForget': tp.forgetFace,
        # rezimi i rucni koraci
        'tomatoModeAuto': lambda: tp.setControlMode('auto'),
        'tomatoModeManual': lambda: tp.setControlMode('manual'),
        'tomatoBucketOn': lambda: tp.setBucketMode(True),
        'tomatoBucketOff': lambda: tp.setBucketMode(False),
        'tomatoAssistantOn': lambda: tp.setAssistant(True),
        'tomatoAssistantOff': lambda: tp.setAssistant(False),
        'tomatoVoiceOn': lambda: tp.setVoice(True),
        'tomatoVoiceOff': lambda: tp.setVoice(False),
        'tomatoPickOne': lambda: tp.startPicking('one'),
        'tomatoManualGrab': lambda: tp.startPicking('grab'),
        'tomatoManualDeliver': lambda: tp.startPicking('deliver'),
        'tomatoManualPlace': lambda: tp.startPicking('place'),
        # ostalo
        'tomatoSettingsReset': tp.resetSettings,
        'tomatoResetStats': tp.resetStats,
    }
    if cmd in simple:
        return simple[cmd]()
    if cmd == 'tomatoPreview':
        tomatoPicker.preview()
    elif cmd == 'tomatoMaskOn':
        tomatoPicker.setShowMask(True)
    elif cmd == 'tomatoMaskOff':
        tomatoPicker.setShowMask(False)
    elif cmd == 'tomatoLogOn':
        tomatoPicker.setLogging(True)
        return 'Snimanje uzoraka za trening uključeno.'
    elif cmd == 'tomatoLogOff':
        tomatoPicker.setLogging(False)
        return 'Snimanje uzoraka isključeno.'
    elif cmd == 'tomatoSaveGrabTarget':
        return tomatoPicker.saveGrabTarget()
    elif cmd == 'tomatoSaveGrabPose':
        return tomatoPicker.saveGrabPose()
    elif cmd == 'tomatoSaveLookPose':
        return tomatoPicker.saveLookPose()
    elif cmd == 'tomatoColorReset':
        return tomatoPicker.resetColor()
    elif cmd == 'tomatoClawTest':
        return tomatoPicker.testClaw()
    elif cmd == 'tomatoSaveClawClosed':
        return tomatoPicker.saveClaw('closed')
    elif cmd == 'tomatoSaveClawOpen':
        return tomatoPicker.saveClaw('open')
    return None


async def check_permit(websocket):
    while True:
        recv_str = await websocket.recv()
        cred_dict = recv_str.split(":")
        if cred_dict[0] == "admin" and cred_dict[1] == "123456":
            response_str = "congratulation, you have connect with server\r\nnow, you can do something else"
            await websocket.send(response_str)
            return True
        else:
            response_str = "sorry, the username or password is wrong, please submit again"
            await websocket.send(response_str)

async def recv_msg(websocket):
    global speed_set, modeSelect
    move.setup()

    while True: 
        response = {
            'status' : 'ok',
            'title' : '',
            'data' : None
        }

        data = ''
        data = await websocket.recv()
        try:
            data = json.loads(data)
        except Exception as e:
            print('not A JSON')

        if not data:
            continue

        # IZMENA: greska u jednoj komandi (npr. pun SD kartica pri snimanju) NE sme
        # da prekine vezu - stranica bi tada izgubila i STOP dok se ponovo poveze
        try:
            if isinstance(data,str):
                robotCtrl(data, response)

                switchCtrl(data, response)

                functionSelect(data, response)

                configPWM(data, response)

                if 'get_info' == data:
                    response['title'] = 'get_info'
                    response['data'] = [info.get_cpu_tempfunc(), info.get_cpu_use(), info.get_ram_info()]

                if 'tomatoStatus' == data:
                    response['title'] = 'tomatoStatus'
                    response['data'] = tomatoPicker.getStatus()

                elif 'tomatoGetSettings' == data:
                    response['title'] = 'tomatoSettings'
                    response['data'] = tomatoPicker.getSettings()

                # IZMENA: kalibracija i podesavanja berbe paradajza (tomato.html)
                elif data.startswith('tomato') and data not in ('tomatoStart', 'tomatoStop'):
                    msg = tomatoCommand(data)
                    if msg:
                        response['title'] = 'tomatoMsg'
                        response['data'] = msg

                if 'wsB' in data:
                    try:
                        set_B=data.split()
                        speed_set = int(set_B[1])
                    except:
                        pass

                elif 'AR' == data:
                    modeSelect = 'AR'
                    screen.screen_show(4, 'ARM MODE ON')
                    try:
                        fpv.changeMode('ARM MODE ON')
                    except:
                        pass

                elif 'PT' == data:
                    modeSelect = 'PT'
                    screen.screen_show(4, 'PT MODE ON')
                    try:
                        fpv.changeMode('PT MODE ON')
                    except:
                        pass

                #CVFL
                elif 'CVFL' == data:
                    flask_app.modeselect('findlineCV')

                elif 'CVFLColorSet' in data:
                    color = int(data.split()[1])
                    flask_app.camera.colorSet(color)

                elif 'CVFLL1' in data:
                    pos = int(data.split()[1])
                    flask_app.camera.linePosSet_1(pos)

                elif 'CVFLL2' in data:
                    pos = int(data.split()[1])
                    flask_app.camera.linePosSet_2(pos)

                elif 'CVFLSP' in data:
                    err = int(data.split()[1])
                    flask_app.camera.errorSet(err)

                elif 'defEC' in data:#Z
                    fpv.defaultExpCom()

            elif(isinstance(data,dict)):
                if data.get('title') == "findColorSet":
                    color = data['data']
                    flask_app.colorFindSet(color[0],color[1],color[2])
                # IZMENA: kalibracija boje paradajza dodirom na sliku - server sam
                # cita boju iz SIROVOG frejma oko dodirnute tacke (tacnije od
                # boje procitane iz JPEG prenosa u pregledacu)
                elif data.get('title') == "tomatoCalib":
                    try:
                        x, y = data['data'][0], data['data'][1]
                        response['title'] = 'tomatoMsg'
                        response['data'] = tomatoPicker.calibrateColorAt(x, y)
                    except Exception as e:
                        response['title'] = 'tomatoMsg'
                        response['data'] = 'Greška pri kalibraciji: %s' % e
                # IZMENA: boja kantice dodirom na sliku
                elif data.get('title') == "bucketCalib":
                    try:
                        x, y = data['data'][0], data['data'][1]
                        response['title'] = 'tomatoMsg'
                        response['data'] = tomatoPicker.calibrateBucketColorAt(x, y)
                    except Exception as e:
                        response['title'] = 'tomatoMsg'
                        response['data'] = 'Greška pri kalibraciji kantice: %s' % e
                # IZMENA: status sa dnevnikom dogadjaja (samo novi dogadjaji)
                elif data.get('title') == "tomatoStatus":
                    response['title'] = 'tomatoStatus'
                    response['data'] = tomatoPicker.getStatus(
                        events_after=(data.get('data') or {}).get('eventsAfter', 0))
                # IZMENA: izmena podesavanja sa taba "Podesavanja" (proverava se
                # svaka vrednost - tip i dozvoljeni opseg, vidi tomato_config.SETTINGS)
                # IZMENA: osobe (prepoznavanje lica po imenu)
                elif data.get('title') in ("faceEnroll", "faceRename", "faceWave", "faceDelete"):
                    d = data.get('data') or {}
                    t = data['title']
                    try:
                        if t == 'faceEnroll':
                            msg = tomatoPicker.enrollFace(name=d.get('name'), pid=d.get('pid'))
                        elif t == 'faceRename':
                            msg = tomatoPicker.renamePerson(str(d.get('pid')), d.get('name'))
                        elif t == 'faceWave':
                            msg = tomatoPicker.setPersonWave(str(d.get('pid')), bool(d.get('on')))
                        else:
                            msg = tomatoPicker.deletePerson(str(d.get('pid')))
                    except Exception as e:
                        msg = 'Greška: %s' % e
                    response['title'] = 'tomatoMsg'
                    response['data'] = msg
                elif data.get('title') == "tomatoSet":
                    ok, msg = tomatoPicker.applySettings(data.get('data'))
                    response['title'] = 'tomatoSettings'
                    response['status'] = 'ok' if ok else 'error'
                    response['data'] = tomatoPicker.getSettings()
                    response['msg'] = msg

            else:
                pass
        except Exception as e:
            print('[WEB] greska pri obradi %r: %s' % (data, e))
            response = {'status': 'error', 'title': 'tomatoMsg', 'data': 'Greška: %s' % e}
        # IZMENA: ne zatrpavaj log porukama koje stizu vise puta u sekundi
        if data not in ('tomatoStatus', 'get_info') and not (isinstance(data, dict) and data.get('title') == 'tomatoStatus'):
            print(data)
        response = json.dumps(response)
        await websocket.send(response)

async def main_logic(websocket, path):
    await check_permit(websocket)
    await recv_msg(websocket)

def show_wlan0_ip():
    try:
        if OLED_connection:
            result = subprocess.run(
                "ifconfig wlan0 | grep 'inet ' | awk '{print $2}'",
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8'
            ) 
            screen.screen_show(2, "IP:" + result.stdout.strip())
    except Exception as e:
        pass

def show_network_mode():
    try:
        if OLED_connection:
            result = subprocess.run(
                "if iw dev wlan0 link | grep -q 'Connected'; then echo 'Station Mode'; else echo 'AP Mode'; fi",
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8'
            )
            screen.screen_show(3, result.stdout.strip())
    except Exception as e:
        pass

if __name__ == '__main__':
    switch.switchSetup()
    switch.set_all_switch_off()

    show_wlan0_ip()
    time.sleep(0.5)
    show_network_mode()

    global flask_app
    flask_app = app.webapp()
    flask_app.startthread()
    ws2812 = robotLight.Adeept_SPI_LedPixel(16, 255)
    try:
        if ws2812.check_spi_state() != 0:
            ws2812.start()
            ws2812.breath(70,70,255)
    except:
        ws2812.led_close()
        pass

    while  1:
        try:                  #Start server,waiting for client
            start_server = websockets.serve(main_logic, '0.0.0.0', 8888)
            asyncio.get_event_loop().run_until_complete(start_server)
            print('waiting for connection...')
            # print('...connected from :', addr)
            break
        except Exception as e:
            print(e)
            ws2812.set_all_led_color_data(0,0,0)
            ws2812.show()

        try:
            ws2812.set_all_led_color_data(0,80,255)
            ws2812.show()
        except:
            pass
    try:
        asyncio.get_event_loop().run_forever()
    except Exception as e:
        print(e)
        ws2812.set_all_led_color_data(0,0,0)
        ws2812.show()
        move.destroy()
