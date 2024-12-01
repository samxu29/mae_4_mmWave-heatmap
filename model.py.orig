import torch
import timm
import numpy as np
import logging
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

logging.basicConfig(level=logging.INFO)

def random_indexes(size: int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches: torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))
        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]
        return patches, forward_indexes, backward_indexes

IMG_SIZE = (640, 480)
PATCH_SIZE = 32
# with 16x16 Patch Size:
#     Number of patches along width: 640/16=40640/16=40
#     Number of patches along height: 480/16=30480/16=30
#     Total number of patches: 40×30=120040×30=1200

# with 32x32 Patch Size:
#     Number of patches along width: 640/32=20640/32=20
#     Number of patches along height: 480/32=15480/32=15
#     Total number of patches: 20×15=30020×15=300

class MAE_Encoder(torch.nn.Module):
    def __init__(self, image_size=IMG_SIZE, patch_size=PATCH_SIZE, emb_dim=192, num_layer=12, num_head=3, mask_ratio=0.75) -> None:
        super().__init__()
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size[0] // patch_size) * (image_size[1] // patch_size), 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)
        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        logging.debug(f"\tEncoder \tInput Image Shape: {img.shape}")
        patches = self.patchify(img) 
        logging.debug(f"\tPatches Shape After Conv2d: {patches.shape}")
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        logging.debug(f"\tPatches Shape After Rearrange: {patches.shape}")
        patches = patches + self.pos_embedding
        logging.debug(f"\tPatches Shape After Adding Positional Embedding: {patches.shape}")
        patches, forward_indexes, backward_indexes = self.shuffle(patches)
        logging.debug(f"\tPatches Shape After Shuffle: {patches.shape}")
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        logging.debug(f"\tPatches Shape After Adding CLS Token: {patches.shape}")
        patches = rearrange(patches, 't b c -> b t c')
        logging.debug(f"\tPatches Shape Before Transformer: {patches.shape}")
        features = self.layer_norm(self.transformer(patches))
        logging.debug(f"\tFeatures Shape After Transformer: {features.shape}")
        features = rearrange(features, 'b t c -> t b c')
        logging.debug(f"\tFeatures Shape After Rearrange: {features.shape}")
        return features, backward_indexes

class MAE_Decoder(torch.nn.Module):
    def __init__(self, image_size=IMG_SIZE, patch_size=PATCH_SIZE, emb_dim=192, num_layer=4, num_head=3) -> None:
        super().__init__()
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size[0] // patch_size) * (image_size[1] // patch_size) + 1, 1, emb_dim))
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size[1] // patch_size)
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        logging.debug(f"\tDecoder \tInput Features Shape: {features.shape}")
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        logging.debug(f"\tFeatures Shape After Adding Mask Token: {features.shape}")
        features = take_indexes(features, backward_indexes)
        logging.debug(f"\tFeatures Shape After Unshuffle: {features.shape}")
        features = features + self.pos_embedding
        logging.debug(f"\tFeatures Shape After Adding Positional Embedding: {features.shape}")
        features = rearrange(features, 't b c -> b t c')
        logging.debug(f"\tFeatures Shape Before Transformer: {features.shape}")
        features = self.transformer(features)
        logging.debug(f"\tFeatures Shape After Transformer: {features.shape}")
        features = rearrange(features, 'b t c -> t b c')
        logging.debug(f"\tFeatures Shape After Rearrange: {features.shape}")
        features = features[1:]  # remove global feature
        logging.debug(f"\tFeatures Shape After Removing CLS Token: {features.shape}")
        patches = self.head(features)
        logging.debug(f"\tPatches Shape After Linear Head: {patches.shape}")
        mask = torch.zeros_like(patches)
        mask[T - 1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        logging.debug(f"\tReconstructed Image Shape: {img.shape}")
        mask = self.patch2img(mask)
        logging.debug(f"\tMask Shape: {mask.shape}")
        return img, mask

class MAE_ViT(torch.nn.Module):
    def __init__(self, image_size=IMG_SIZE, patch_size=PATCH_SIZE, emb_dim=192, encoder_layer=12, encoder_head=3, decoder_layer=4, decoder_head=3, mask_ratio=0.75) -> None:
        super().__init__()
        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features, backward_indexes)
        return predicted_img, mask

class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder: MAE_Encoder, num_classes=10) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, img):
        logging.debug(f"\tClassifier \tInput Image Shape: {img.shape}")
        patches = self.patchify(img)
        logging.debug(f"\tPatches Shape After Conv2d: {patches.shape}")
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        logging.debug(f"\tPatches Shape After Rearrange: {patches.shape}")
        patches = patches + self.pos_embedding
        logging.debug(f"\tPatches Shape After Adding Positional Embedding: {patches.shape}")
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        logging.debug(f"\tPatches Shape After Adding CLS Token: {patches.shape}")
        patches = rearrange(patches, 't b c -> b t c')
        logging.debug(f"\tPatches Shape Before Transformer: {patches.shape}")
        features = self.layer_norm(self.transformer(patches))
        logging.debug(f"\tFeatures Shape After Transformer: {features.shape}")
        features = rearrange(features, 'b t c -> t b c')
        logging.debug(f"\tFeatures Shape After Rearrange: {features.shape}")
        logits = self.head(features[0])
        logging.debug(f"\tLogits Shape: {logits.shape}")
        return logits

if __name__ == '__main__':
    batch_size = 13
    img = torch.rand((batch_size, 3, IMG_SIZE[0], IMG_SIZE[1]))

    logging.debug('----------------------------------------------------------')
    logging.debug('---- MAE Encoder:')
    logging.debug(f'\tINPUT: {img.shape}\n')

    encoder = MAE_Encoder()
    features, backward_indexes = encoder(img)

    logging.debug(f'\n\t\tOUTPUT: {features.shape}')
    logging.debug('----------------------------------------------------------')

    logging.debug('---- MAE Decoder:')
    logging.debug(f'\tINPUT: {features.shape}\n')

    decoder = MAE_Decoder()
    predicted_img, mask = decoder(features, backward_indexes)

    logging.debug(f'\n\t\tOUTPUT: {predicted_img.shape}, {mask.shape}')
    logging.debug('----------------------------------------------------------')

    logging.debug('---- ViT Classifier:')
    logging.debug(f'\tINPUT: {img.shape}\n')

    classifier = ViT_Classifier(encoder)
    logits = classifier(img)

    logging.debug(f'\n\t\tOUTPUT: {logits.shape}')
    logging.debug('----------------------------------------------------------')
