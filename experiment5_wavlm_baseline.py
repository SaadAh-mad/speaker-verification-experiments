import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

from transformers import (
    WavLMModel,
    Wav2Vec2FeatureExtractor
)





DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#################################################
# ASTP
#################################################




#################################################
# Processor
#################################################

processor = Wav2Vec2FeatureExtractor.from_pretrained(
    "microsoft/wavlm-base"
)

#################################################
# Model
#################################################

class WavLMSpeakerModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.wavlm = WavLMModel.from_pretrained(
            "microsoft/wavlm-base"
        )

    def forward(self, input_values):

        outputs = self.wavlm(
        input_values=input_values,
        output_hidden_states=True
    )

        hidden_states = outputs.hidden_states

        hidden = 0

        for h in hidden_states:
            hidden += h

        #hidden = hidden / len(hidden_states)

        #embedding = hidden.mean(dim=1)
        hidden = sum(outputs.hidden_states) / 13
        embedding = hidden.mean(dim=1)

        embedding = F.normalize(
        embedding,
        dim=-1
    )

        return embedding

#################################################
# Load Model
#################################################

model = WavLMSpeakerModel().to(DEVICE)
#model.load_state_dict(
    #torch.load(
        #"experiment4_astp_model.pth",
       # map_location=DEVICE
    #)
#)

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
    padding=True
)

    input_values = features.input_values.to(
    DEVICE
)

    with torch.no_grad():

        embedding = model(
    input_values
)

    return embedding


#################################################
# TEST FILES
#################################################

same_1 = "mini_voxceleb1/train/id11132-OG_Pzr-UXdM-00006.wav"
same_2 = "mini_voxceleb1/train/id11132-gM9BuqYleR4-00004.wav"

diff_1 = "mini_voxceleb1/train/id10938-sTDdgK3bGlQ-00005.wav"
diff_2 = "mini_voxceleb1/train/id10910-AwBMp1w8KKQ-00004.wav"

#################################################
# SAME SPEAKER
#################################################

emb1 = get_embedding(same_1)
emb2 = get_embedding(same_2)

#rint("\nEmbedding 1 shape:")
#print(emb1.shape)

#print("\nFirst 20 values of Embedding 1:")
#print(emb1[0][:20])

#print("\nEmbedding 2 shape:")
#print(emb2.shape)

#print("\nFirst 20 values of Embedding 2:")
#print(emb2[0][:20])

same_score = F.cosine_similarity(
    emb1,
    emb2
).item()


#################################################
# DIFFERENT SPEAKER
#################################################

emb3 = get_embedding(diff_1)
emb4 = get_embedding(diff_2)

#print("\nEmbedding 3 shape:")
#print(emb3.shape)

#print("\nFirst 20 values of Embedding 3:")
#print(emb3[0][:20])

#print("\nEmbedding 4 shape:")
#print(emb4.shape)

#print("\nFirst 20 values of Embedding 4:")
#print(emb4[0][:20])

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