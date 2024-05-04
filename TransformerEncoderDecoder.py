import torch
import torch.nn as nn
import torchvision.models as models

class ImagePatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        patches = self.projection(x).permute(0, 2, 3, 1).view(batch_size, -1, self.projection.out_channels)
        return patches

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=2048, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.position_embeddings = nn.Parameter(torch.randn(1, 1 + (256 // 16) ** 2, embed_dim))

    def forward(self, patches):
        batch_size, num_patches, _ = patches.shape
        patches = patches + self.position_embeddings[:, 1:].repeat(1, batch_size, 1)
        patches = self.dropout(patches)
        encoded_patches = self.transformer_encoder(patches)
        return encoded_patches

class Encoder(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=768, num_heads=12, num_layers=12):
        super(Encoder, self).__init__()
        self.patch_embedding = ImagePatchEmbedding(in_channels, patch_size, embed_dim)
        self.transformer_encoder = TransformerEncoder(embed_dim, num_heads, num_layers)

    def forward(self, images):
        patches = self.patch_embedding(images)
        encoded_patches = self.transformer_encoder(patches)
        return encoded_patches
    
class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, max_seq_length=20):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_seq_length, embed_size)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(embed_size, hidden_size) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, captions, encoder_out):
        batch_size = captions.size(0)
        caption_embeddings = self.word_embeddings(captions)
        position_embeddings = self.position_embeddings(torch.arange(0, captions.size(1)).unsqueeze(0).repeat(batch_size, 1))
        embeddings = caption_embeddings + position_embeddings
        encoder_out = encoder_out.unsqueeze(0)
        for layer in self.layers:
            embeddings = layer(embeddings, encoder_out)
        out = self.fc_out(embeddings)
        return out