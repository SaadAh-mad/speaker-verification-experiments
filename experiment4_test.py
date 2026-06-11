import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

from transformers import WhisperProcessor, WhisperModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small"
)


#################################################
# Model
#################################################

class WhisperSpeakerModel(nn.Module):

    def __init__(self, num_speakers):

        super().__init__()

        self.whisper = WhisperModel.from_pretrained(
            "openai/whisper-small"
        )

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

model = WhisperSpeakerModel(
    num_speakers=30
).to(DEVICE)

model.load_state_dict(
    torch.load(
        "experiment4_astp_model.pth",
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded!")


#################################################
# Embedding Extraction
#################################################

def get_embedding(audio_path):

    audio, _ = librosa.load(
        audio_path,
        sr=16000
    )

    features = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding="max_length",
        max_length=480000,
        truncation=True
    )

    input_features = features.input_features.to(
        DEVICE
    )

    with torch.no_grad():

        _, embedding = model(
            input_features
        )

    return embedding


#################################################
# TEST FILES
#################################################

same_1 = "mini_voxceleb1/train/id10012-Mki-3pJgdMw-00008.wav"
same_2 = "mini_voxceleb1/train/id10012-o-8xXV5MC8I-00002.wav"

diff_1 = "mini_voxceleb1/train/id11160-XgiGzJQOHwU-00016.wav"
diff_2 = "mini_voxceleb1/train/id10977-yfbPlRseDPU-00020.wav"

#################################################
# SAME SPEAKER
#################################################

emb1 = get_embedding(same_1)
emb2 = get_embedding(same_2)

print("\nEmbedding 1 shape:")
print(emb1.shape)

print("\nFirst 20 values of Embedding 1:")
print(emb1[0][:20])

print("\nEmbedding 2 shape:")
print(emb2.shape)

print("\nFirst 20 values of Embedding 2:")
print(emb2[0][:20])

same_score = F.cosine_similarity(
    emb1,
    emb2
).item()


#################################################
# DIFFERENT SPEAKER
#################################################

emb3 = get_embedding(diff_1)
emb4 = get_embedding(diff_2)

print("\nEmbedding 3 shape:")
print(emb3.shape)

print("\nFirst 20 values of Embedding 3:")
print(emb3[0][:20])

print("\nEmbedding 4 shape:")
print(emb4.shape)

print("\nFirst 20 values of Embedding 4:")
print(emb4[0][:20])

different_score = F.cosine_similarity(
    emb3,
    emb4
).item()


#################################################
# RESULTS
#################################################

print("\nRESULTS\n")

print(
    f"Same Speaker Score: {same_score:.4f}"
)

print(
    f"Different Speaker Score: {different_score:.4f}"
)

print(
    f"Gap: {same_score - different_score:.4f}"
)