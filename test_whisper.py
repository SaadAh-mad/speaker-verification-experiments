import librosa
import torch
from transformers import WhisperProcessor, WhisperModel

audio, sr = librosa.load("sample.mp3", sr=16000)

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperModel.from_pretrained("openai/whisper-tiny")

inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

with torch.no_grad():
    outputs = model.encoder(inputs.input_features)

print(outputs.last_hidden_state.shape)