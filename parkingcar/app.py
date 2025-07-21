from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import cv2
from ultralytics import solutions
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

MODEL_PATH = "runs/detect/train/weights/best.pt"
JSON_PATH = "output.json"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    video_url = None
    if request.method == 'POST':
        if 'video' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['video']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Process video
        output_path, stats = process_video(filepath)
        video_url = url_for('download_file', filename=os.path.basename(output_path))
        result = stats
    return render_template('index.html', result=result, video_url=video_url)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

@app.route('/video/<filename>')
def serve_video(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), mimetype='video/mp4')

def process_video(input_path):
    cap = cv2.VideoCapture(input_path)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"output_{os.path.basename(input_path)}")
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    parkingmanager = solutions.ParkingManagement(
        model=MODEL_PATH,
        json_file=JSON_PATH,
    )
    total_slots = 0
    available_slots = 0
    frame_count = 0
    while cap.isOpened():
        ret, im0 = cap.read()
        if not ret:
            break
        results = parkingmanager(im0)
        # Thống kê số chỗ trống và tổng số chỗ trên frame này
        if hasattr(results, 'free_spots') and hasattr(results, 'total_spots'):
            available_slots += results.free_spots
            total_slots += results.total_spots
        video_writer.write(results.plot_im)
        frame_count += 1
    cap.release()
    video_writer.release()
    # Tính trung bình số chỗ trống và tổng số chỗ
    avg_total = total_slots // frame_count if frame_count else 0
    avg_free = available_slots // frame_count if frame_count else 0
    stats = {
        'total_slots': avg_total,
        'available_slots': avg_free,
        'occupied_slots': avg_total - avg_free if avg_total > 0 else 0
    }
    return output_path, stats

if __name__ == '__main__':
    app.run(debug=True)
