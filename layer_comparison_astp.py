import librosa
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import WhisperProcessor, WhisperModel

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small"
)

model = WhisperModel.from_pretrained(
    "openai/whisper-small"
)

# Selecting Audio Files
file1 = "mini_voxceleb1/train/id11160-HO06Y9-2pss-00001.wav"
file2 = "mini_voxceleb1/train/id10189-KIse4_aC6FY-00003.wav"

class ASTP(nn.Module):
    """ Attentive statistics pooling: Channel- and context-dependent
        statistics pooling, first used in ECAPA_TDNN.
    """

    def __init__(self,
                 in_dim,
                 bottleneck_dim=128,
                 global_context_att=False,
                 **kwargs):
        super(ASTP, self).__init__()
        self.in_dim = in_dim
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear, then we don't
        # need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(
                in_dim * 3, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(
                in_dim, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim,
                                 kernel_size=1)  # equals V and k in the paper

    def forward(self, x):
        """
        x: a 3-dimensional tensor in tdnn-based architecture (B,F,T)
            or a 4-dimensional tensor in resnet architecture (B,C,F,T)
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        assert len(x.shape) == 3

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(
                torch.var(x, dim=-1, keepdim=True) + 1e-7).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! ReLU may be hard to converge.
        alpha = torch.tanh(
            self.linear1(x_in))  # alpha = F.relu(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(var.clamp(min=1e-7))
        return torch.cat([mean, std], dim=1)

    def get_out_dim(self):
        self.out_dim = 2 * self.in_dim
        return self.out_dim
    
astp = ASTP(in_dim=768)
astp.eval()


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
    hidden = hidden.transpose(1,2)

    
    embedding = astp(hidden)

    embedding = F.normalize(
    embedding,
    dim=-1
)

    return embedding

for layer in range(13):
    emb1 = get_layer_embedding(file1, layer)
    emb2 = get_layer_embedding(file2, layer)

    
    similarity = F.cosine_similarity(
        emb1,
        emb2
    )

    print(
        f"Layer {layer}: {similarity.item():.4f}"
    )