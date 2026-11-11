# Notice and attribution

This project builds on the **Adeept PiCar-Pro** robot kit software:
https://github.com/adeept/Adeept_PiCar-Pro. All rights to the original Adeept code stay with Adeept.

## Adeept files included here in modified form
| File | What I changed |
|---|---|
| `Server/WebServer.py` | WebSocket commands for the tomato panel: picking, calibration, settings, faces, assistant |
| `Server/camera_opencv.py` | new `tomato` CV mode that runs the tomato and bucket detectors on every frame and logs telemetry |
| `Server/RPIservo.py` | single shared PCA9685/I2C connection plus a lock, so threads never write to the bus at the same time |
| `Server/Move.py` | sets the PCA9685 frequency only once (the original reset the chip on every motor command) |
| `Server/OLED.py` | SH1106 display driver instead of SSD1306 |

To run the robot you also need the rest of the original Adeept `Server/` folder (`base_camera.py`, `PID.py`, `Functions.py`, `RobotLight.py` and others). See the README.

## Original work by Adnan Malagić
`TomatoPicker.py`, `tomato_detector.py`, `tomato_classifier.py`, `tomato_config.py`, `tomato_datalog.py`, `tomato_probe.py`, `test_tomato_detector.py`, `bucket_detector.py`, `face_module.py`, `rebuild_faces.py`, `assistant.py`, `dist/tomato.html`, `install_robot.sh`, plus the trained models `tomato_model.json` and `tomato_clf_weights.json`.

## Third-party models (not included)
YuNet and SFace come from OpenCV Zoo (Apache 2.0). The Haar cascade comes from OpenCV.
