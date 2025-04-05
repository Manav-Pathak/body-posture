from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from video.video_processing import process_video  # Updated import (moved into 'video' folder)
from audio.audio_analysis import extract_audio, analyze_audio

app = Flask(__name__)

# Configure upload and processed folders
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Global variable to store final report data for the final report page
final_report_data = {}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/live-analysis')
def live_analysis():
    return "<h2 style='text-align:center; padding:40px;'>Live Analysis (Coming Soon)</h2>"

@app.route('/video-upload', methods=['GET'])
def video_upload():
    return render_template('upload.html')

@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # # Save the uploaded video file
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(upload_path)

    # # Define processed file name and path for video
    processed_filename = "processed_" + file.filename
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)

    #------------------TOTAL-----------------

    # Process video and get posture report
    try:
        posture_report = process_video(upload_path, processed_path)
    except Exception as e:
        return f"Error processing video: {str(e)}", 500

    #-----------------------------------------------
    #Audio Extraction and Analysis
    
    audio_filename = "extracted_" + file.filename.rsplit('.', 1)[0] + ".mp3"
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
    
    try:
        extract_audio(upload_path, audio_path)
        audio_analysis_result = analyze_audio(audio_path)
    except Exception as e:
        audio_analysis_result = {"error": str(e)}

    #------------------TOTAL-----------------

    #------------------ONLY AUDIO-----------------
    # audio_path = "uploads/extracted_wtsp_fast.mp3" 
    
    # try:
    #     audio_analysis_result = analyze_audio(audio_path)
    # except Exception as e:
    #     audio_analysis_result = {"error": str(e)}
    
    # posture_report = {"counts": {}, "final_posture": "N/A", "total_frames_counted": 0}  #placeholder

    #------------------ONLY AUDIO-----------------

    # Store final report data globally (for /final-report route)
    global final_report_data
    final_report_data = {
        "posture": posture_report,
        "audio": audio_analysis_result
    }
    
    # Render the download page with processed video and audio analysis summary.
    return render_template('download.html', filename=processed_filename, audio_analysis=audio_analysis_result)

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

@app.route('/final_report')
def final_report():
    global final_report_data
    return render_template('final_report.html', data=final_report_data)

if __name__ == '__main__':
    app.run(debug=True)
