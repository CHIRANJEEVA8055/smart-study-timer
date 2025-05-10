from flask import Flask, render_template, Response, jsonify
import cv2
import time
import csv
from datetime import datetime, timedelta

app = Flask(__name__)
cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

study_time = 0
last_check = time.time()
timer_running = False

log_file = "study_log.csv"

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    global study_time, last_check, timer_running
    while True:
        success, frame = cam.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            current_time = time.time()
            if len(faces) > 0:
                if not timer_running:
                    last_check = current_time
                    timer_running = True
                else:
                    study_time += current_time - last_check
                    last_check = current_time
                label = "Focused"
                color = (0, 255, 0)
            else:
                timer_running = False
                label = "Away"
                color = (0, 0, 255)

            formatted = str(timedelta(seconds=int(study_time)))
            cv2.putText(frame, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Study Time: {formatted}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_time')
def get_time():
    formatted = str(timedelta(seconds=int(study_time)))
    return jsonify(time=formatted)

@app.route('/save')
def save_log():
    with open(log_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), int(study_time)])
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(debug=True)
