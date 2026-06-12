import librosa
import torch
import torch.nn.functional as F

from transformers import (
    WavLMModel,
    Wav2Vec2FeatureExtractor
)

processor = Wav2Vec2FeatureExtractor.from_pretrained(
    "microsoft/wavlm-base"
)

model = WavLMModel.from_pretrained(
    "microsoft/wavlm-base"
)

# Selecting Audio Files
file1 = "mini_voxceleb1/train/id10012-GQxAiL_gSJg-00014.wav"
file2 = "mini_voxceleb1/train/id10012-o-8xXV5MC8I-00004.wav"

def get_embedding(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=16000
    )

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(
            inputs.input_values,
            output_hidden_states=True
        )

    hidden = outputs.last_hidden_state

    embedding = hidden.mean(dim=1)

    embedding = F.normalize(
        embedding,
        dim=-1
    )

    return embedding

emb1 = get_embedding(file1)
emb2 = get_embedding(file2)

score = F.cosine_similarity(
    emb1,
    emb2
)

print(score.item())