from moviepy.editor import VideoFileClip
from pydub import AudioSegment, silence

def extract_audio(video_path, audio_path):
    """
    Extracts audio from the given video file and writes it to audio_path.
    """
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, codec='mp3')
    clip.close()

def analyze_audio(audio_path):
    """
    Analyzes the audio file for long pauses and determines speech speed.
    Returns a dict with:
      - num_long_pauses: number of pauses longer than min_silence_len
      - total_silence_s: total silence duration in ms
      - silence_ratio: fraction of audio that is silence
      - speech_speed: categorized as 'Too Slow', 'Proper', or 'Too Fast'
    """
    # Load audio using pydub
    audio = AudioSegment.from_file(audio_path)
    
    # Parameters: detect pauses longer than 1500ms; threshold is relative to average loudness
    min_silence_len = 1500  
    silence_thresh = audio.dBFS - 16  # adjust as needed
    
    # Detect silent intervals; returns list of [start, end] in ms
    silences = silence.detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    
    num_long_pauses = len(silences)

    total_silence =  sum((end - start) for start, end in silences)

    total_duration = len(audio)  
    silence_ratio = total_silence / total_duration if total_duration > 0 else 0

   
    if silence_ratio > 0.4:
        speech_speed = "Too Slow"
    elif silence_ratio < 0.2:
        speech_speed = "Too Fast"
    else:
        speech_speed = "Proper"

    return {
        "num_long_pauses": num_long_pauses,
        "total_silence_s": total_silence / 1000,
        #"silence_ratio": silence_ratio,
        "speech_speed": speech_speed
    }

# ---------------------------------------------------------------------------
# Testing snippet (for direct audio testing without video input)
# Uncomment these lines to test audio analysis independently.

if __name__ == "__main__":
    test_audio_path = "uploads/extracted_wtsp_fast.mp3"
    result = analyze_audio(test_audio_path)
    print("Test Audio Analysis Result:", result)
