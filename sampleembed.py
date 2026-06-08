import librosa
import torch
from transformers import WhisperProcessor, WhisperModel
import torch.nn.functional as F

print("Loading Whisper...")

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperModel.from_pretrained("openai/whisper-tiny")

audio, sr = librosa.load("sample.mp3", sr=16000)

inputs = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model.encoder(inputs.input_features)

hidden = outputs.last_hidden_state

print("Encoder Output Shape:")
print(hidden.shape)

embedding = hidden.mean(dim=1)

print("Speaker Embedding Shape:")
print(embedding.shape)

print("First 10 values:")
print(embedding[0][:10])

sim = F.cosine_similarity(
    embedding,
    embedding
)

print(sim)