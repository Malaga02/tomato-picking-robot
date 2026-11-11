#!/usr/bin/env/python
# File name   : camera_opencv.py
# Website     : www.Adeept.com
# Author      : Adeept
# Date		  : 2025/03/12
import cv2
from base_camera import BaseCamera
import RPIservo
import numpy as np
# IZMENA: nova detekcija cheri paradajza (boja sa "obavijanjem" crvene oko
# 0/180, vise pragova zasicenosti, ML ocena kandidata, pracenje kroz vreme) -
# vidi tomato_detector.py. Radi u posebnom CV rezimu 'tomato' koji NE pomera
# servo motore (servoe tada kontrolise samo TomatoPicker.py) - ranije su i
# CV nit i TomatoPicker istovremeno pomerali isti pan/tilt servo (kanal 1 i
# 3), pa bi kamera "skocila" nazad cim se paradajz pojavi i odmah ga izgubila.
import tomato_detector
import tomato_config
import tomato_datalog
import bucket_detector   # IZMENA: zelena kantica za odlaganje paradajza
import Move as move
import Switch as switch
import datetime
import Kalman_Filter as Kalman_filter
import PID
import time
import threading
import imutils

import picamera2
import libcamera

from picamera2 import Picamera2
import io
import subprocess
import re


def face_module_ascii(name):
    try:
        import face_module
        return face_module.ascii_name(name)
    except Exception:
        return ''


def find_usb_camera_indices():
    # IZMENA: USB veb-kamera ne dobija uvek isti /dev/videoN cvor. Pi ima
    # ugradjene ISP/HEVC uredjaje (bcm2835-isp, rpi-hevc-dec) koji zauzimaju
    # neke brojeve cvorova cak i bez prikljucene CSI kamere, a redosled
    # otkrivanja USB uredjaja pri svakom pokretanju/boot-u nije uvek isti.
    # Zbog toga kamera ume da "iskoci" na /dev/video0 jednom, a drugi put
    # na /dev/video1 ili /dev/video2 - fiksni indeks 0 tada ne postoji ili
    # pokazuje na pogresan (ISP) uredjaj, pa kamera ne moze da se pokrene
    # ("Could not start camera") ili se povremeno zamrzne. Ova funkcija
    # trazi stvarne /dev/videoN cvorove USB kamere po njenom nazivu/putanji
    # (isti nalaz kao "v4l2-ctl --list-devices"), umesto da pretpostavlja
    # fiksni broj.
    try:
        output = subprocess.check_output(
            ['v4l2-ctl', '--list-devices'], stderr=subprocess.STDOUT,
            text=True, timeout=5
        )
    except Exception as e:
        print(f"find_usb_camera_indices: v4l2-ctl --list-devices failed: {e}")
        return []

    indices = []
    for block in output.strip().split('\n\n'):
        lines = [l for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        header = lines[0].lower()
        if 'usb' in header or 'webcam' in header or 'camera' in header:
            for line in lines[1:]:
                m = re.search(r'/dev/video(\d+)', line)
                if m:
                    indices.append(int(m.group(1)))
    return indices


def open_usb_camera():
    # IZMENA: prvo probaj cvorove pronadjene preko find_usb_camera_indices()
    # (stvarna USB kamera po nazivu), pa tek onda indekse 0-3 kao rezervu,
    # umesto da se odmah oslanja na fiksni indeks 0.
    candidates = find_usb_camera_indices()
    for i in (0, 1, 2, 3):
        if i not in candidates:
            candidates.append(i)

    for index in candidates:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"open_usb_camera: using /dev/video{index}")
                return cap
        cap.release()
    return None

Dv = -1 #Directional variable

pid = PID.PID()
pid.SetKp(0.5)
pid.SetKd(0)
pid.SetKi(0)

Threshold = 80 
findLineMove = 1
tracking_servo_status = 0
FLCV_Status = 0

CVRun = 1
linePos_1 = 440
linePos_2 = 380
lineColorSet = 255
frameRender = 1
findLineError = 20
turn_speed = 35 
forward_speed = 20 

APPMode = None
hflip = 0 
vflip = 0 
ImgIsNone = 0

colorUpper = np.array([44, 255, 255])
colorLower = np.array([24, 100, 100])

def map(input, in_min,in_max,out_min,out_max):
    return (input-in_min)/(in_max-out_min)*(out_max-out_min)+out_min

class CVThread(threading.Thread):
    font = cv2.FONT_HERSHEY_SIMPLEX

    kalman_filter_X =  Kalman_filter.Kalman_filter(0.01,0.1)
    kalman_filter_Y =  Kalman_filter.Kalman_filter(0.01,0.1)
    P_direction = -1
    T_direction = -1
    P_servo = 1 
    T_servo = 3 
    P_anglePos = 0
    T_anglePos = 0
    cameraDiagonalW = 64
    cameraDiagonalH = 48
    # IZMENA: kamera sada hvata frejmove u 320x240 (vidi frames() ispod -
    # rezolucija je ranije smanjena sa 640x480 radi CPU/napajanja), pa
    # videoW/videoH MORAJU da prate stvarnu rezoluciju. Ako ostanu na starim
    # 640x480, i brzina centriranja (stepen po pikselu, ispod u servoMove())
    # i sama tacka centra kadra (koju findColor() koristi) su pogresni - to
    # je bio uzrok da robot "promasuje" paradajz iako je prijavljivao da je
    # centriran/uhvacen.
    videoW = 320
    videoH = 240
    Y_lock = 0
    X_lock = 0
    tor = 17

    scGear = RPIservo.ServoCtrl()
    scGear.moveInit()
    Tracking_sc = RPIservo.ServoCtrl()
    Tracking_sc.start()
    move.setup()

    def __init__(self, *args, **kwargs):
        self.CVThreading = 0
        self.CVMode = 'none'
        self.imgCV = None

        self.mov_x = None
        self.mov_y = None
        self.mov_w = None
        self.mov_h = None

        self.radius = 0
        self.box_x = None
        self.box_y = None
        self.drawing = 0
        self.imgT = 0.0   # IZMENA: vreme snimanja frejma (za TomatoPicker)

        self.findColorDetection = 0

        self.left_Pos1 = None
        self.right_Pos1 = None
        self.center_Pos1 = None

        self.left_Pos2 = None
        self.right_Pos2 = None
        self.center_Pos2 = None

        self.center = None
        
        self.tracking_servo_left = None
        self.tracking_servo_left_mark = 0
        self.tracking_servo_right_mark = 0
        self.servo_left_stop = 0
        self.servo_right_stop = 0

        super(CVThread, self).__init__(*args, **kwargs)
        self.__flag = threading.Event()
        self.__flag.clear()

        self.avg = None
        self.motionCounter = 0
        self.lastMovtionCaptured = datetime.datetime.now()
        self.frameDelta = None
        self.thresh = None
        self.cnts = None

    def mode(self, invar, imgInput, t=None):
        self.CVMode = invar
        self.imgCV = imgInput
        self.imgT = time.time() if t is None else t
        self.resume()

    def elementDraw(self,imgInput):
        if self.CVMode == 'none':
            pass

        elif self.CVMode == 'findColor':
            if self.findColorDetection:
                cv2.putText(imgInput,'Target Detected',(40,60), CVThread.font, 0.5,(255,255,255),1,cv2.LINE_AA)
                self.drawing = 1
            else:
                cv2.putText(imgInput,'Target Detecting',(40,60), CVThread.font, 0.5,(255,255,255),1,cv2.LINE_AA)
                self.drawing = 0

            if self.radius > 10 and self.drawing:
                cv2.rectangle(imgInput,(int(self.box_x-self.radius),int(self.box_y+self.radius)),(int(self.box_x+self.radius),int(self.box_y-self.radius)),(255,255,255),1)

        elif self.CVMode == 'tomato':
            imgInput = Camera.tomato_det.draw(imgInput, show_mask=Camera.tomato_show_mask)
            Camera.bucket_det.draw(imgInput)
            # IZMENA: zona hvatanja - gde paradajz treba da bude (u pozi
            # hvatanja) da bi bio tacno izmedju prstiju hvataljke
            gz = Camera.grab_zone
            if gz is not None:
                h, w = imgInput.shape[:2]
                gx, gy, gr = int(gz[0] * w), int(gz[1] * h), max(4, int(gz[2] * w))
                col = (255, 0, 255) if gz[3] else (0, 0, 255)
                for a in range(0, 360, 30):
                    cv2.ellipse(imgInput, (gx, gy), (gr, gr), 0, a, a + 15, col, 1, cv2.LINE_AA)
                cv2.putText(imgInput, 'hvataljka' if gz[3] else 'hvataljka (NIJE kalibrisano)',
                            (max(0, gx - gr), min(h - 4, gy + gr + 14)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, col, 1, cv2.LINE_AA)

        elif self.CVMode == 'findlineCV':
            if frameRender:
                imgInput = cv2.cvtColor(imgInput, cv2.COLOR_BGR2GRAY)
                retval_bw, imgInput =  cv2.threshold(imgInput, Threshold, 255, cv2.THRESH_BINARY) 
                imgInput = cv2.erode(imgInput, None, iterations=2) 
                imgInput = cv2.dilate(imgInput, None, iterations=2) 

            try:
                if lineColorSet == 255:
                    cv2.putText(imgInput,('Following White Line'),(30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(128,255,128),1,cv2.LINE_AA)
                else:
                    cv2.putText(imgInput,('Following Black Line'),(30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(128,255,128),1,cv2.LINE_AA)
                imgInput=cv2.merge((imgInput.copy(),imgInput.copy(),imgInput.copy()))
                cv2.line(imgInput,(self.left_Pos1,(linePos_1+30)),(self.left_Pos1,(linePos_1-30)),(255,128,64),2)
                cv2.line(imgInput,(self.right_Pos1,(linePos_1+30)),(self.right_Pos1,(linePos_1-30)),(64,128,255),2)
                cv2.line(imgInput,(0,linePos_1),(640,linePos_1),(255,128,64),1)

                cv2.line(imgInput,(self.left_Pos2,(linePos_2+30)),(self.left_Pos2,(linePos_2-30)),(64,128,255),2)
                cv2.line(imgInput,(self.right_Pos2,(linePos_2+30)),(self.right_Pos2,(linePos_2-30)),(64,128,255),2)
                cv2.line(imgInput,(0,linePos_2),(640,linePos_2),(64,128,255),1)

                cv2.line(imgInput,((self.center-20),int((linePos_1+linePos_2)/2)),((self.center+20),int((linePos_1+linePos_2)/2)),(0,0,0),1)
                cv2.line(imgInput,((self.center),int((linePos_1+linePos_2)/2+20)),((self.center),int((linePos_1+linePos_2)/2-20)),(0,0,0),1)

            except:
                pass

        elif self.CVMode == 'watchDog':
            if self.drawing:
                cv2.rectangle(imgInput, (self.mov_x, self.mov_y), (self.mov_x + self.mov_w, self.mov_y + self.mov_h), (128, 255, 0), 1)

        return imgInput


    def watchDog(self, imgInput):
        timestamp = datetime.datetime.now()
        gray = cv2.cvtColor(imgInput, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.avg is None:
            self.avg = gray.copy().astype("float")
            return 'background model'

        cv2.accumulateWeighted(gray, self.avg, 0.5)
        self.frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg))
        self.thresh = cv2.threshold(self.frameDelta, 5, 255,
            cv2.THRESH_BINARY)[1]
        self.thresh = cv2.dilate(self.thresh, None, iterations=2)
        self.cnts = cv2.findContours(self.thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        self.cnts = imutils.grab_contours(self.cnts)
        for c in self.cnts:
            if cv2.contourArea(c) < 5000:
                continue
            (self.mov_x, self.mov_y, self.mov_w, self.mov_h) = cv2.boundingRect(c)
            self.drawing = 1
            
            self.motionCounter += 1
            self.lastMovtionCaptured = timestamp

        if (timestamp - self.lastMovtionCaptured).seconds >= 0.5:
            self.drawing = 0
        self.pause()



    def findLineCtrl(self, posInput):
        global findLineMove,tracking_servo_status,FLCV_Status

        
        if FLCV_Status == 0:    
            CVThread.scGear.moveAngle(0, 0) 
            CVThread.scGear.moveAngle(1, 0)

            FLCV_Status = 1
        if posInput != None and findLineMove == 1:
            if FLCV_Status == -1:
                CVThread.Tracking_sc.stopWiggle()
                self.tracking_servo_left_mark = 0
                self.tracking_servo_right_mark = 0
                FLCV_Status = 1
            if posInput > 480: 
                tracking_servo_status = 1 
                if CVRun:
                    CVThread.scGear.moveAngle(0, 30) 
                    time.sleep(0.15)
                    move.video_Tracking_Move(turn_speed, 1) 
                else:
                    CVThread.scGear.moveAngle(0, 0)
                    move.motorStop()

            elif posInput < 180: 
                tracking_servo_status = -1 
                if CVRun:
                    CVThread.scGear.moveAngle(0, -30)
                    time.sleep(0.15)
                    move.video_Tracking_Move(turn_speed, 1)
                
                else:
                    CVThread.scGear.moveAngle(0, 0)
                    move.motorStop() 
                        
            else:
                tracking_servo_status = 0 
                if CVRun:
                    CVThread.scGear.moveAngle(0, 0)
                    time.sleep(0.15)
                    move.video_Tracking_Move(turn_speed, 1) 
                else: 
                    move.motorStop()
                pass
        
        else:
            move.motorStop() 
            FLCV_Status = -1
            if tracking_servo_status == -1 :
                CVThread.scGear.moveAngle(0, 30)
                time.sleep(0.15)
                move.video_Tracking_Move(turn_speed, 1)

            elif tracking_servo_status == 1 : 
                CVThread.scGear.moveAngle(0, -30)
                time.sleep(0.15)
                move.video_Tracking_Move(turn_speed, 1)

            else: 
                pass

    def findlineCV(self, frame_image):
        frame_findline = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)

        retval, frame_findline =  cv2.threshold(frame_findline, Threshold, 255, cv2.THRESH_BINARY)
        frame_findline = cv2.erode(frame_findline, None, iterations=2)
        frame_findline = cv2.dilate(frame_findline, None, iterations=2)
        colorPos_1 = frame_findline[linePos_1]
        colorPos_2 = frame_findline[linePos_2]
        
        try:
            lineColorCount_Pos1 = np.sum(colorPos_1 == lineColorSet)
            lineColorCount_Pos2 = np.sum(colorPos_2 == lineColorSet)

            lineIndex_Pos1 = np.where(colorPos_1 == lineColorSet)
            lineIndex_Pos2 = np.where(colorPos_2 == lineColorSet)


            if lineIndex_Pos1 !=[]:
                if abs(lineIndex_Pos1[0][-1] - lineIndex_Pos1[0][0]) > 500:
                    print("Tracking color not found")
                    findLineMove = 0 
                else:
                    findLineMove = 1
            elif lineIndex_Pos2!= []:
                if abs(lineIndex_Pos2[0][-1] - lineIndex_Pos2[0][0]) > 500:
                    print("Tracking color not found")
                    findLineMove = 0
                else:
                    findLineMove = 1

            if lineColorCount_Pos1 == 0:
                lineColorCount_Pos1 = 1
            if lineColorCount_Pos2 == 0:
                lineColorCount_Pos2 = 1

            self.left_Pos1 = lineIndex_Pos1[0][1] 
            self.right_Pos1 = lineIndex_Pos1[0][lineColorCount_Pos1-2]   # 

            self.center_Pos1 = int((self.left_Pos1+self.right_Pos1)/2)

            self.left_Pos2 =  lineIndex_Pos2[0][1]
            self.right_Pos2 = lineIndex_Pos2[0][lineColorCount_Pos2-2]
            self.center_Pos2 = int((self.left_Pos2+self.right_Pos2)/2)

            self.center = int((self.center_Pos1+self.center_Pos2)/2)
        except:
            self.center = None
            pass

        self.findLineCtrl(self.center)
        self.pause()


    def servoMove(ID, Dir, errorInput):
        if ID == 1:
            errorGenOut = CVThread.kalman_filter_X.kalman(errorInput)
            CVThread.P_anglePos += 0.15*(errorGenOut*Dir)*CVThread.cameraDiagonalW/CVThread.videoW

            if abs(errorInput) > CVThread.tor:
                CVThread.scGear.moveAngle(ID,CVThread.P_anglePos)
                CVThread.X_lock = 0
            else:
                CVThread.X_lock = 1
        elif ID == 3:
            errorGenOut = CVThread.kalman_filter_Y.kalman(errorInput)
            CVThread.T_anglePos += 0.1*(errorGenOut*Dir)*CVThread.cameraDiagonalH/CVThread.videoH

            if abs(errorInput) > CVThread.tor:
                CVThread.scGear.moveAngle(ID,CVThread.T_anglePos)
                CVThread.Y_lock = 0
            else:
                CVThread.Y_lock = 1
        else:
            print('No servoPort %d assigned.'%ID)

    def findColor(self, frame_image):
        if APPMode == 'APP':
            hsv = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
        else:
            hsv = cv2.cvtColor(frame_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, colorLower, colorUpper)#1
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        # (Originalni Adeept "Find Color" rezim sa glavne stranice - za berbu
        # paradajza se koristi poseban rezim 'tomato', vidi tomato_detector.py.)
        if len(cnts) > 0:
            self.findColorDetection = 1
            c = max(cnts, key=cv2.contourArea)
            ((self.box_x, self.box_y), self.radius) = cv2.minEnclosingCircle(c)
            X = int(self.box_x)
            Y = int(self.box_y)
            # IZMENA: centar kadra iz STVARNE rezolucije (videoW/videoH)
            error_Y = CVThread.videoH // 2 - Y
            error_X = CVThread.videoW // 2 - X
            CVThread.servoMove(CVThread.P_servo, CVThread.P_direction, -error_X)
            CVThread.servoMove(CVThread.T_servo, CVThread.T_direction, -error_Y)
        else:
            self.findColorDetection = 0
        self.pause()


    def pause(self):
        self.__flag.clear()

    def resume(self):
        self.__flag.set()

    def run(self):
        while 1:
            self.__flag.wait()
            if self.CVMode == 'none':
                continue
            
            elif self.CVMode == 'findColor':
                self.CVThreading = 1
                self.findColor(self.imgCV)
                self.CVThreading = 0
            elif self.CVMode == 'tomato':
                self.CVThreading = 1
                try:
                    # IZMENA: prvo kantica - paradajz koji se vidi u njoj (vec
                    # ubran) se iskljucuje iz detekcije
                    b = Camera.bucket_det.process(self.imgCV, t=self.imgT, ignore=Camera.bucket_ignore)
                    # samo dok robot TRAZI paradajz - inace bi i paradajz koji
                    # robot drzi u hvataljci (ispred kantice) bio "iskljucen"
                    if b['detected'] and Camera.picker_state in ('', 'IDLE', 'TRAZENJE', 'CENTRIRANJE', 'PRILAZAK'):
                        m = 0.08 * b['bw']
                        Camera.tomato_det.exclude_boxes = [(b['x'] - b['bw'] / 2 - m, b['top'] - m,
                                                            b['x'] + b['bw'] / 2 + m, b['bottom'] - 0.05 * b['bh'])]
                    else:
                        Camera.tomato_det.exclude_boxes = []
                except Exception as e:
                    print('[BUCKET] greska u detekciji: %s' % e, flush=True)
                try:
                    res = Camera.tomato_det.process(self.imgCV, t=self.imgT)
                    Camera.datalog.maybe_log(self.imgCV, res, Camera.picker_state)
                except Exception as e:
                    print('[TOMATO] greska u detekciji: %s' % e, flush=True)
                self.CVThreading = 0
                self.pause()
            elif self.CVMode == 'findlineCV':
                self.CVThreading = 1
                self.findlineCV(self.imgCV)
                self.CVThreading = 0
            elif self.CVMode == 'watchDog':
                self.CVThreading = 1
                self.watchDog(self.imgCV)
                self.CVThreading = 0
            else:
                pass

class Camera(BaseCamera):
    video_source = 0
    modeSelect = 'none'
    cv_thread = None  # IZMENA: postavlja se u frames() - vidi TomatoPicker.py
    # IZMENA: detektor paradajza, poslednji SIROVI frejm (za kalibraciju boje
    # dodirom na sliku) i snimac podataka za treniranje
    _tcfg = tomato_config.load()
    tomato_det = tomato_detector.TomatoDetector(_tcfg.get('detector'))
    tomato_show_mask = False
    datalog = tomato_datalog.DataLogger()
    datalog.configure(enabled=_tcfg['picker'].get('log_enabled', False),
                      interval=_tcfg['picker'].get('log_interval_s', 2.0),
                      max_frames=_tcfg['picker'].get('log_max_frames', 800))
    bucket_det = bucket_detector.BucketDetector(_tcfg.get('bucket'))
    face_rec = None      # FaceRecognizer - postavlja TomatoPicker (oznake lica na videu)
    picker_state = ''
    bucket_ignore = None # zaklonjeni deo kadra (prsti hvataljke, paradajz koji robot nosi)
    grab_zone = None     # (x_frac, y_frac, r_frac, kalibrisano) - postavlja TomatoPicker
    last_raw = None
    last_raw_t = 0.0


    @staticmethod
    def _face_overlay_due():
        fr = Camera.face_rec
        if fr is None:
            return False
        r = fr.latest()
        return fr.enrolling or (r is not None and time.time() - r['t'] < 1.0)

    @staticmethod
    def _draw_faces(img):
        fr = Camera.face_rec
        txt = None
        if fr.enrolling:
            n, tot = fr.enroll_progress
            txt = 'UCIM LICE: %s %d/%d' % (face_module_ascii(fr.enroll_name), n, tot)
        fr.draw(img, fr.latest(), txt)

    def colorFindSet(self, invarH, invarS, invarV):
        global colorUpper, colorLower
        HUE_1 = invarH+15
        HUE_2 = invarH-15
        if HUE_1>180:HUE_1=180
        if HUE_2<0:HUE_2=0

        SAT_1 = invarS+150
        SAT_2 = invarS-150
        if SAT_1>255:SAT_1=255
        if SAT_2<0:SAT_2=0

        VAL_1 = invarV+150
        VAL_2 = invarV-150
        if VAL_1>255:VAL_1=255
        if VAL_2<0:VAL_2=0

        colorUpper = np.array([HUE_1,SAT_1,VAL_1])
        colorLower = np.array([HUE_2,SAT_2,VAL_2])
        print('HSV_1:%d %d %d'%(HUE_1,SAT_1,VAL_1))
        print('HSV_2:%d %d %d'%(HUE_2,SAT_2,VAL_2))
        print(colorUpper)
        print(colorLower)

    def colorFindSetApp(self, invarH, invarS, invarV):
        global colorUpper, colorLower
        HUE_1 = invarH+100
        HUE_2 = invarH-100
        if HUE_1>255:HUE_1=255
        if HUE_2<0:HUE_2=0

        SAT_1 = invarS+100
        SAT_2 = invarS-100
        if SAT_1>255:SAT_1=255
        if SAT_2<0:SAT_2=0

        VAL_1 = invarV+100
        VAL_2 = invarV-100
        if VAL_1>255:VAL_1=255
        if VAL_2<0:VAL_2=0

        colorUpper = np.array([HUE_1, SAT_1, VAL_1])
        colorLower = np.array([HUE_2, SAT_2, VAL_2])
        print('HSV_1:%d %d %d'%(HUE_1, SAT_1, VAL_1))
        print('HSV_2:%d %d %d'%(HUE_2, SAT_2, VAL_2))
        print(colorUpper)
        print(colorLower)

    def modeSet(self, invar):
        Camera.modeSelect = invar

    def CVRunSet(self, invar):
        global CVRun
        CVRun = invar

    def linePosSet_1(self, invar):
        global linePos_1
        linePos_1 = invar

    def linePosSet_2(self, invar):
        global linePos_2
        linePos_2 = invar

    def colorSet(self, invar):
        global lineColorSet
        lineColorSet = invar

    def randerSet(self, invar):
        global frameRender
        frameRender = invar

    def errorSet(self, invar):
        global findLineError
        findLineError = invar

    def Threshold(self, value):
        global Threshold
        Threshold = value
        
    def ThresholdOK(self):
        global Threshold
        return Threshold

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source


# _______________________________________________________
    @staticmethod
    def frames():
        # --- IZMENA: USB veb-kamera (cv2.VideoCapture) umesto Picamera2 (CSI kamera) ---
        # Picamera2 radi samo sa Raspberry Pi CSI kamerom (ravni kabl). Ako je kamera
        # obicna USB veb-kamera, pristupa joj se preko OpenCV-a, ne preko picamera2.
        global ImgIsNone,hflip,vflip
        # IZMENA: umesto fiksnog indeksa 0 (koji ume da "odseli" na drugi
        # /dev/videoN cvor u zavisnosti od redosleda detekcije uredjaja pri
        # pokretanju), sada se stvarni cvor USB kamere trazi automatski -
        # vidi open_usb_camera()/find_usb_camera_indices() iznad.
        cap = open_usb_camera()
        # IZMENA: smanjena rezolucija (bilo 640x480) da se rastereti CPU/napajanje
        # dok kamera radi istovremeno sa servo motorima i motorima tockova
        if cap is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            # IZMENA: bafer od 1 frejma - inace V4L2 cuva vise starih frejmova
            # pa robot reaguje na sliku staru i po nekoliko stotina ms (pan
            # servo tada "prebaci" metu i klati se levo-desno).
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        if cap is None or not cap.isOpened():
            raise RuntimeError('Could not start camera. Check whether the USB webcam is connected (try: v4l2-ctl --list-devices).')

        # IZMENA: mnoge USB veb-kamere TIHO ignorisu cap.set(...) zahtev za
        # rezolucijom ako je ne podrzavaju tacno (npr. vrate se na svoju
        # podrazumevanu rezoluciju - cesto 640x480 ili vecu) - cv2 NE javlja
        # gresku u tom slucaju. CVThread.videoW/videoH su ranije bili
        # hardkodirani na 320/240 PRETPOSTAVLJAJUCI da je cap.set() uspeo, pa
        # su SVI proracuni centriranja (findColor/servoMove, kao i
        # TomatoPicker._steerTowardTarget) racunali "centar kadra" na 160x120
        # dok je stvarni frejm bio siri/visi - centriranje je zbog toga
        # sistematski gadjalo pogresnu tacku i NIKAD nije uspevalo da se
        # spusti ispod tolerancije (tor=17 / STEER_READY_TOLERANCE=12), pa je
        # CENTRIRANJE ostajala zaglavljena zauvek. Zato ovde citamo STVARNU
        # rezoluciju koju je kamera prihvatila (cap.get) i njom overwrite-ujemo
        # CVThread.videoW/videoH, umesto da se oslanjamo na zahtevanu vrednost.
        actualW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 320
        actualH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 240
        if actualW != 320 or actualH != 240:
            print('[CAMERA] UPOZORENJE: kamera ignorisala trazenih 320x240, '
                  'stvarna rezolucija je %dx%d - koristim STVARNU vrednost '
                  'za centriranje.' % (actualW, actualH), flush=True)
        CVThread.videoW = actualW
        CVThread.videoH = actualH

        cvt = CVThread()
        cvt.start()
        # IZMENA: instanca CVThread-a se cuva na klasu Camera da bi joj drugi
        # moduli (npr. TomatoPicker.py) mogli da pristupe iz svoje niti radi
        # citanja stanja detekcije boje (findColorDetection, radius, X_lock,
        # Y_lock) bez dupliranja CV logike.
        Camera.cv_thread = cvt

        while True:
            start_time = time.time()
            ret, img = cap.read()
            t_cap = time.time()
            if ret and img is not None:
                if hflip:
                    img = cv2.flip(img, 1)
                if vflip:
                    img = cv2.flip(img, 0)
            else:
                img = None

            if img is None:
                if ImgIsNone == 0:
                    print("--------------------")
                    print("\033[31merror: Unable to read camera data.\033[0m")
                    print("\033[33mIt may be that the Legacy camera is not turned on or the camera is not connected correctly.\033[0m")
                    print("Use the command: \033[34m'raspistill -t 1000 -o image.jpg'\033[0m to check whether the camera can be used correctly.")
                    print("Press the keyboard keys \033[34m'Ctrl + C'\033[0m multiple times to exit the current program.")
                    print("--------Ctrl+C quit-----------")
                    ImgIsNone = 1
                # IZMENA: kratka pauza - bez nje ova petlja 100% zauzme jezgro
                # dok kamera ne vraca frejmove
                time.sleep(0.05)
                continue
            Camera.last_raw = img
            Camera.last_raw_t = t_cap
            if Camera.modeSelect == 'none':
                cvt.pause()
            else:
                # IZMENA: CV nit dobija KOPIJU frejma. Ranije je elementDraw()
                # crtao tekst i pravougaonik po ISTOM nizu piksela koji je CV
                # nit u tom trenutku analizirala - beli okvir oko paradajza je
                # tako ulazio u samu analizu slike.
                if not cvt.CVThreading:
                    try:
                        cvt.mode(Camera.modeSelect, img.copy(), t_cap)
                    except Exception:
                        pass
                try:
                    img = cvt.elementDraw(img.copy())
                except Exception:
                    pass
            # IZMENA: oznake prepoznatih lica (ako je pozdrav vlasnika
            # ukljucen ili traje ucenje lica) - u svakom rezimu kamere
            if Camera._face_overlay_due():
                try:
                    if img is Camera.last_raw:
                        img = img.copy()   # sirovi frejm (za analizu) mora ostati bez oznaka
                    Camera._draw_faces(img)
                except Exception:
                    pass
            # IZMENA: nizi JPEG kvalitet (bilo podrazumevano ~95) da se smanji
            # opterecenje CPU-a pri kompresiji svakog frejma
            jpeg_quality = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
            # IZMENA: JPEG kodiranje JEDNOM po frejmu (original je pozivao
            # imencode dva puta - duplo trosenje CPU-a za svaki frejm)
            ok, buf = cv2.imencode('.jpg', img, jpeg_quality)
            if ok:
                yield buf.tobytes()
            
