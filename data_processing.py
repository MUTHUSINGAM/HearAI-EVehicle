import io
import librosa
import numpy as np
import soundfile as sf

def process_audio_file(fileobj, target_sr=16000, n_mels=64, n_mfcc=40):
    '''
    Reads an uploaded file-like object, resamples to target_sr, converts to mono,
    and returns a log-mel spectrogram (or optionally MFCC for downstream ML input).
    '''
    # Read from uploaded file object or local path
    if hasattr(fileobj, "read"):
        y, sr = sf.read(io.BytesIO(fileobj.read()))
    else:
        y, sr = sf.read(fileobj)

    # Mono conversion
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    # Resample
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Return mono waveform as float32 for YAMNet
    y = y.astype(np.float32)
    return y
    # Optionally extract features if needed elsewhere:
    # S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    # log_S = librosa.power_to_db(S, ref=np.max)
    # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # return log_S
