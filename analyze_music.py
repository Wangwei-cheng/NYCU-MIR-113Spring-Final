import subprocess
import librosa
import numpy as np
import os

def separate_audio(input_path):
    print("使用 Demucs 進行音源分離...")
    subprocess.run(["demucs", input_path], check=True)

def analyze_instruments(input_path, demucs_output_folder="separated/htdemucs/output"):
    separate_audio(input_path)
    
    parts = {
        "drums": "鼓",
        "bass": "貝斯",
        "vocals": "人聲",
        "other": "其他樂器（如吉他、鋼琴、合成器）"
    }

    detected = []
    for part, desc in parts.items():
        path = os.path.join(demucs_output_folder, f"{part}.wav")
        y, sr = librosa.load(path, sr=None)
        rms = np.mean(librosa.feature.rms(y=y))
        if rms > 0.01:  # 門檻值，可微調
            detected.append(desc)

    return detected

def analyze_tempo(input_path):
    y, sr = librosa.load(input_path, sr=None)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    return tempo

if __name__ == "__main__":
    instruments = analyze_instruments("output.mp3")
    print("偵測到的樂器：", instruments)
    tempo = analyze_tempo("output.mp3")
    print("tempo：", tempo)