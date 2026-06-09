import librosa
import torch
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperModel

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-tiny"
)

model = WhisperModel.from_pretrained(
    "openai/whisper-tiny"
)

# Selecting Audio Files
file1 = "mini_voxceleb1/train/id10012-0AXjxNXiEzo-00001.wav"
file2 = "mini_voxceleb1/train/id10189-S9cB6BBQy-w-00001.wav"

def get_layer_embedding(audio_path,layer_num):
    audio,sr = librosa.load(audio_path, sr=16000)

    inputs  = processor(
        audio,
        sampling_rate = 16000,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model.encoder(
            inputs.input_features,
            output_hidden_states=True
        )

    hidden = outputs.hidden_states[layer_num]

    embedding = hidden.mean(dim=1)

    return embedding

for layer in range(5):
    emb1 = get_layer_embedding(file1, layer)
    emb2 = get_layer_embedding(file2, layer)

    similarity = F.cosine_similarity(
        emb1,
        emb2
    )

    print(
        f"Layer {layer}: {similarity.item():.4f}"
    )