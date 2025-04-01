from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from video_processing import process_video  # Import the video processing function
from audio.audio_analysis import extract_audio, analyze_audio  # Import audio extraction & analysis functions

app = Flask(__name__)

# Configure upload and processed folders
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/live-analysis')
def live_analysis():
    # Placeholder
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

    # Save the uploaded file
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(upload_path)

    # Define processed file name and path for the video
    processed_filename = "processed_" + file.filename
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)

    # Process video (drawing landmarks & classification)
    try:
        process_video(upload_path, processed_path)
    except Exception as e:
        return f"Error processing video: {str(e)}", 500

    audio_filename = "extracted_" + file.filename.rsplit('.', 1)[0] + ".mp3"
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
    
    try:
        # Extract 
        extract_audio(upload_path, audio_path)
        # Analyze 
        audio_analysis_result = analyze_audio(audio_path)
    except Exception as e:
        audio_analysis_result = {"error": str(e)}

    return render_template('download.html', filename=processed_filename, audio_analysis=audio_analysis_result)

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
