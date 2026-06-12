import os
import glob
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor, WhisperModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

        return audio, label


#################################################
# Collate Function
#################################################

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small"
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
    padding="max_length",
    max_length=480000,
    truncation=True
)

    #print(features.input_features.shape)
    return features.input_features, labels


#################################################
# Model
#################################################

class WhisperSpeakerModel(nn.Module):

    def __init__(self, num_speakers):

        super().__init__()

        self.whisper = WhisperModel.from_pretrained(
            "openai/whisper-small"
        )

        # freeze Whisper
        for p in self.whisper.parameters():
            p.requires_grad = False

        # 13 layers
        self.layer_weights = nn.Parameter(
            torch.ones(13)
        )

        self.pooling = ASTP(768)

        self.classifier = nn.Linear(
        1536,
        num_speakers
)

    def forward(self, input_features):

        outputs = self.whisper.encoder(
            input_features,
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

        # Mean Pooling
        combined = combined.transpose(1, 2)

        embedding = self.pooling(
        combined
)

        logits = self.classifier(
            embedding
        )

        return logits, embedding


#################################################
# Main
#################################################

dataset = MiniVoxDataset(
    "mini_voxceleb1/train"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

model = WhisperSpeakerModel(
    num_speakers=len(dataset.spk2idx)
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    [
        model.layer_weights,
        *model.pooling.parameters(),
        *model.classifier.parameters()
    ],
    lr=1e-3
)

#################################################
# Training
#################################################

for epoch in range(5):

    total_loss = 0

    for features, labels in loader:

        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        logits, embeddings = model(features)

        loss = criterion(
            logits,
            labels
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1} "
        f"Loss = {total_loss:.4f}"
    )

print("\nLearned Layer Weights:\n")

print(
    torch.softmax(
        model.layer_weights,
        dim=0
    )
)

torch.save(
    model.state_dict(),
    "experiment4_astp_model.pth"
)

print("ASTP model saved!")