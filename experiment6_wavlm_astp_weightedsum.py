import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from torch.utils.data import Dataset, DataLoader
import os
import gc
import glob
from transformers import (
    WavLMModel,
    Wav2Vec2FeatureExtractor
)
import random
import numpy as np
import glob
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

#################################################
# ASTP
#################################################

class ASTP(nn.Module):

    def __init__(self, in_dim, bottleneck_dim=128):
        super().__init__()

        self.linear1 = nn.Conv1d(
            in_dim,
            bottleneck_dim,
            kernel_size=1
        )

        self.linear2 = nn.Conv1d(
            bottleneck_dim,
            in_dim,
            kernel_size=1
        )

    def forward(self, x):

        alpha = torch.tanh(
            self.linear1(x)
        )

        alpha = torch.softmax(
            self.linear2(alpha),
            dim=2
        )

        mean = torch.sum(
            alpha * x,
            dim=2
        )

        var = torch.sum(
            alpha * (x ** 2),
            dim=2
        ) - mean ** 2

        std = torch.sqrt(
            var.clamp(min=1e-7)
        )

        return torch.cat(
            [mean, std],
            dim=1
        )


#################################################
# Processor
#################################################

processor = Wav2Vec2FeatureExtractor.from_pretrained(
    "microsoft/wavlm-base"
)

def collate_fn(batch):

    audios = [x[0] for x in batch]

    labels = torch.tensor(
        [x[1] for x in batch]
    )

    features = processor(
        audios,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    return features.input_values, labels

#################################################
# Dataset
#################################################

class MiniVoxDataset(Dataset):

    def __init__(self, root):

        self.files = sorted(
            glob.glob(os.path.join(root, "*.wav"))
        )

        speakers = sorted(
            list(
                set(
                    os.path.basename(f).split("-")[0]
                    for f in self.files
                )
            )
        )

        self.spk2idx = {
            spk: idx
            for idx, spk in enumerate(speakers)
        }

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        wav_path = self.files[idx]

        speaker_id = os.path.basename(
            wav_path
        ).split("-")[0]

        label = self.spk2idx[speaker_id]

        audio, _ = librosa.load(
        wav_path,
        sr=16000
)

        audio = audio[:48000]

        return audio, label

dataset = MiniVoxDataset(
    "/content/mini_voxceleb1/train"
)

print("Files:", len(dataset))
print("Speakers:", len(dataset.spk2idx))

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)



#################################################
# Model
#################################################

class WavLMSpeakerModel(nn.Module):

    def __init__(self, num_speakers):

        super().__init__()

        self.wavlm = WavLMModel.from_pretrained(
        "microsoft/wavlm-base"
)
        

        self.layer_weights = nn.Parameter(
            torch.ones(13)
        )

        for p in self.wavlm.parameters():
          p.requires_grad = False

        self.pooling = ASTP(768)

        self.classifier = nn.Linear(
            1536,
            num_speakers
        )

    def forward(self, input_values):

        with torch.no_grad():
          outputs = self.wavlm(
          input_values=input_values,
          output_hidden_states=True
    )
        hidden_states = outputs.hidden_states

        weights = torch.softmax(
            self.layer_weights,
            dim=0
        )

        combined = 0

        for w, h in zip(weights, hidden_states):
            combined = combined + w * h

        combined = combined.transpose(1, 2)

        embedding = self.pooling(
            combined
        )

        embedding = F.normalize(
            embedding,
            dim=-1
        )

        logits = self.classifier(
            embedding
        )

        return logits, embedding


#################################################
# Load Model
#################################################

model = WavLMSpeakerModel(
    num_speakers=len(dataset.spk2idx)
).to(DEVICE)

model.load_state_dict(
        torch.load(
        "/content/drive/MyDrive/wavlm_astp_model.pth",
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded!")

def get_embedding(audio_path):

    audio, _ = librosa.load(
        audio_path,
        sr=16000
    )

    audio = audio[:48000]

    features = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    input_values = features.input_values.to(
        DEVICE
    )

    with torch.no_grad():

        _, embedding = model(
            input_values
        )

    return embedding


speaker_files = {}

all_files = glob.glob(
    "/content/mini_voxceleb1/train/*.wav"
)

for f in all_files:

    speaker = os.path.basename(f).split("-")[0]

    if speaker not in speaker_files:
        speaker_files[speaker] = []

    speaker_files[speaker].append(f)


same_scores = []

for _ in range(1000):

    speaker = random.choice(
        list(speaker_files.keys())
    )

    files = speaker_files[speaker]

    if len(files) < 2:
        continue

    file1, file2 = random.sample(
        files,
        2
    )

    emb1 = get_embedding(file1)
    emb2 = get_embedding(file2)

    score = F.cosine_similarity(
        emb1,
        emb2
    ).item()

    same_scores.append(score)

different_scores = []

speakers = list(
    speaker_files.keys()
)

for _ in range(1000):

    spk1, spk2 = random.sample(
        speakers,
        2
    )

    file1 = random.choice(
        speaker_files[spk1]
    )

    file2 = random.choice(
        speaker_files[spk2]
    )

    emb1 = get_embedding(file1)
    emb2 = get_embedding(file2)

    score = F.cosine_similarity(
        emb1,
        emb2
    ).item()

    different_scores.append(score)

#ACCURACY COMPUTE USING 0.42 as threshold
threshold = 0.42

correct_same = sum(
    s > threshold
    for s in same_scores
)

correct_diff = sum(
    s < threshold
    for s in different_scores
)

accuracy = (
    correct_same +
    correct_diff
) / (
    len(same_scores) +
    len(different_scores)
)

print("\nRESULTS\n")

print(
    f"Average Same Speaker Score: "
    f"{np.mean(same_scores):.4f}"
)

print(
    f"Average Different Speaker Score: "
    f"{np.mean(different_scores):.4f}"
)

print(
    f"Gap: "
    f"{np.mean(same_scores)-np.mean(different_scores):.4f}"
)

print(
    f"Accuracy: {accuracy:.4f}"
)

print("\nSame Speaker Statistics")
print("Min:", np.min(same_scores))
print("Max:", np.max(same_scores))
print("Std:", np.std(same_scores))

print("\nDifferent Speaker Statistics")
print("Min:", np.min(different_scores))
print("Max:", np.max(different_scores))
print("Std:", np.std(different_scores))
