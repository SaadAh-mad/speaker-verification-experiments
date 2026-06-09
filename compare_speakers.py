import os
import librosa
import torch
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperModel

#Loading Whisper Foundation Model (Frozen)

print("Loading Whisper...")
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperModel.from_pretrained("openai/whisper-tiny")

# Selecting Audio Files
file1 = "mini_voxceleb1/train/id10086-tN8q2rhCDec-00002.wav"
file2 = "mini_voxceleb1/train/id11132-OG_Pzr-UXdM-00014.wav"

# ============================================================
# Extract Speaker Embedding
#
# Pipeline:
# Audio
#   ↓
# Whisper Encoder
#   ↓
# Frame-level Representation (1500 × 384)
#   ↓
# Mean Pooling
#   ↓
# Speaker Embedding (384-D)
# ============================================================

def get_embedding(audio_path):

    audio, sr = librosa.load(audio_path, sr=16000)

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model.encoder(inputs.input_features)

    embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding

# Generating Embeddings

emb1 = get_embedding(file1)
emb2 = get_embedding(file2)

#Displaying Embeddings

print("\nEmbedding 1 shape:")
print(emb1.shape)

print("\nFirst 20 values of Embedding 1:")
print(emb1[0][:20])

print("\nEmbedding 2 shape:")
print(emb2.shape)

print("\nFirst 20 values of Embedding 2:")
print(emb2[0][:20])

#Comparing Embeddings using Cosine Similarity

similarity = F.cosine_similarity(
    emb1,
    emb2
)

print("\nCosine Similarity:")
print(similarity.item())