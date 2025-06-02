import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import requests

instrument_keywords = [
    "singing", "instrument", "guitar", "piano", "keyboard", "organ", "synthesizer", "harpsichord",
    "drum", "cymbal", "hi-hat", "tambourine", "maraca", "gong", "bells", "xylophone",
    "vibraphone", "steelpan", "horn", "trumpet", "trombone", "violin", "fiddle", "cello",
    "double bass", "flute", "saxophone", "clarinet", "harp", "harmonica", "accordion",
    "bagpipes", "didgeridoo", "theremin", "banjo", "mandolin", "ukulele", "zither",
    "sitar", "tabla", "wood block", "rattle", "bell", "chime", "timpani", "sampler",
    "scratching", "orchestra"
]

genre_keywords = [
    "Pop music", "Hip hop music", "Beatboxing", "Rock music", "Heavy metal", "Punk rock", "Grunge", "Progressive rock",
    "Rock and roll", "Psychedelic rock", "Rhythm and blues", "Soul music", "Reggae", "Country", "Swing music",
    "Bluegrass", "Funk", "Folk music", "Middle Eastern music", "Jazz", "Disco", "Classical music", "Opera",
    "Electronic music", "House music", "Techno", "Dubstep", "Drum and bass", "Electronica", "Electronic dance music",
    "Ambient music", "Trance music", "Music of Latin America", "Salsa music", "Flamenco", "Blues", "Music for children",
    "New-age music", "Vocal music", "A capella", "Music of Africa", "Afrobeat", "Christian music", "Gospel music",
    "Music of Asia", "Carnatic music", "Music of Bollywood", "Ska", "Traditional music", "Independent music",
    "Theme music", "Jingle (music)", "Soundtrack music", "Lullaby", "Video game music", "Christmas music",
    "Dance music", "Wedding music", "Happy music", "Sad music", "Tender music", "Exciting music", "Angry music", "Scary music"
]

yamnet_model = None
yamnet_labels = None

def init_yamnet():
    global yamnet_model, yamnet_labels

    if yamnet_model is None:
        print("載入 YAMNet 模型...")
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

    if yamnet_labels is None:
        print("載入 YAMNet 標籤...")
        url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
        response = requests.get(url)
        lines = response.text.strip().split('\n')[1:]
        yamnet_labels = [line.split(',')[2].strip() for line in lines]


def yamnet_analyze(input_path):
    if yamnet_model is None or yamnet_labels is None:
        raise RuntimeError("請先呼叫 init_yamnet()")

    waveform, sr = librosa.load(input_path, sr=16000, mono=True)
    scores, embeddings, spectrogram = yamnet_model(waveform)

    mean_scores = tf.reduce_mean(scores, axis=0).numpy()

    nonzero_indices = np.where(mean_scores > 0.001)[0]
    results = [(yamnet_labels[i], mean_scores[i]) for i in nonzero_indices]
    results.sort(key=lambda x: x[1], reverse=True)

    instruments = [label for label, score in results if any(k in label.lower() for k in instrument_keywords)]
    genres = [label for label, score in results if any(k in label for k in genre_keywords)]
    genres = genres[:5]

    return instruments, genres

def tempo_analyze(input_path):
    y, sr = librosa.load(input_path, sr=None)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    return tempo

if __name__ == "__main__":
    init_yamnet()
    instruments, genres = yamnet_analyze("output.mp3")
    print("樂器：", instruments)
    print("曲風：", genres)
    tempo = tempo_analyze("output.mp3")
    print("tempo：", tempo)