import librosa
import torch

from transformers import WhisperProcessor, WhisperModel

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-tiny"
)

model = WhisperModel.from_pretrained(
    "openai/whisper-tiny"
)

audio, sr = librosa.load(
    "mini_voxceleb1/train/id10086-tN8q2rhCDec-00002.wav",
    sr=16000
)

inputs = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
)

with torch.no_grad():

    outputs = model.encoder(
        inputs.input_features,
        output_hidden_states=True
    )

print("Number of hidden states:")
print(len(outputs.hidden_states))

for i, hidden in enumerate(outputs.hidden_states):
    print(
        f"Layer {i}: {hidden.shape}"
    )