import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

def random_indexes(size : int):
    forward_indexes = np.arange(size)   #   this creates a evenly spaced numpy array
    np.random.shuffle(forward_indexes)  #   shuffels the forward index array on the first dimension
    backward_indexes = np.argsort(forward_indexes)  #   last axis is sorted for the forward index and backward index
    return forward_indexes, backward_indexes    #   this give two random np arrays which one is shuffled and one is sorted

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))#   sequences arranges in the dimension of repeat function
    #   torch.gather - it takes the elements from specific dimension(in this case it is the first)
    #   and it try to get the element respective to the index position 
    #   the repeat function just repeates the elements in the sequence last dimension and adds it to 3 dimension of the indexs

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio #    idk

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape #   patches is a three dimensional shape and it could be time(ig), batch and channel
        remain_T = int(T * (1 - self.ratio))#   the remain_T is the int of size of first dimension minus the ratio

        indexes = [random_indexes(T) for _ in range(B)]#    it just loop for the size of second dimension and generates a random tensor
        
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)# it gets all the first elements(forward_indes variable) of the array index and stacks them into one tensor and sends it to where the patches tensor is
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)#     the same but for the backward_index

        patches = take_indexes(patches, forward_indexes)#   it takes the patches and shuffels them
        patches = patches[:remain_T]#   it just gets the the first ratio patches along the first dimension

        return patches, forward_indexes, backward_indexes # this returns the patches along with forward index and backward index 

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))#     it creates a np arrays full of zeros of shape(1,1,192)
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))#  (256,1,192)
        self.shuffle = PatchShuffle(mask_ratio)#    creates a patch ig

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)#     apply 2 dimensional convolution on what idk

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        #   torch.nn.squencential = stacks multiple layer in the in a sequential manner
        #   creates a list of blocks (which are a part of neural network) emd_dim represents the embedded dimension or the number of io features
        #   and it creates 12 layers of these blocks

        self.layer_norm = torch.nn.LayerNorm(emb_dim)#  applies layer normalization on the embeded

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)#  applies truncated normalized distribution  to mask_token
        trunc_normal_(self.pos_embedding, std=.02)#  same as the last one

    def forward(self, img):
        patches = self.patchify(img)#   applies 2d convolution to the image
        patches = rearrange(patches, 'b c h w -> (h w) b c')#   then it rearranges the tensor from batch, channel, height, width to (height*width),batch and channel
        patches = patches + self.pos_embedding #     does element wise addition with patches and position embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)#  then it calls the forward function in shuffle class

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)#  so it expands the class token to the patch size 
        patches = rearrange(patches, 't b c -> b t c')#  then it reshapes the patches by switching the first and the second dimension
        features = self.layer_norm(self.transformer(patches))#   then applies layer normalization
        features = rearrange(features, 'b t c -> t b c')#   then it is reshaped again into the inital shape

        return features, backward_indexes#   then the features are retuned with backward index

class MAE_Decoder(torch.nn.Module):
    
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim)) #   it creates a np arrays full of zeros of shape(1,1,192)
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))#  (256,1,192)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        #   torch.nn.squencential = stacks multiple layer in the in a sequential manner
        #   creates a list of blocks (which are a part of neural network) emd_dim represents the embedded dimension or the number of io features
        #   and it creates 12 layers of these blocks

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2) # applies linear transformation to the incoming data 
        #   embedding is in_feature and patch_size is the out_feature
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)#   then it reshapes the tensor from
        #   (height * width) batch (channel * patch1 * patch2)
        #   to
        #   batch, channel, (height * patch1), (width * patch2)

        self.init_weight()#  initalizes weights

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)#     applies truncated normalized distribution  to mask_token
        trunc_normal_(self.pos_embedding, std=.02)#  does the same thing but for positional embedding

    def forward(self, features, backward_indexes):
        T = features.shape[0]#  takes the time dimension (ig) but images do not have time dimension
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)#     adds zeros to a to the backward index
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        #   first it matches masked tokens size to backward index to the first dimension of the 
        #   backward feature minus the first dimension of the feature  
        #   and also to the backward index of the feature 
        features = take_indexes(features, backward_indexes) #   it applies the take_indexes function to feature tensor
        features = features + self.pos_embedding #   does a element wise addition to features

        features = rearrange(features, 't b c -> b t c')#   it swaps the first and the second dimension in the features tensor
        features = self.transformer(features)#   features tensor is passed through a transformer 
        features = rearrange(features, 'b t c -> t b c')#   it swaps second and the second dimension in the features tensor
        features = features[1:] #   remove global feature

        patches = self.head(features)#   then apply linear transformation to the feature tensor
        mask = torch.zeros_like(patches)#   it gives the same tensor filled with zeros of the shape of the mask
        mask[T-1:] = 1 #    from the feature first dimension to the end it is filled with ones 
        mask = take_indexes(mask, backward_indexes[1:] - 1) #   the take_indexes function is passed
        img = self.patch2img(patches)#  then applies patches2images function to patches
        mask = self.patch2img(mask)#    then applies patches2images function to mask

        return img, mask #   it returns the images and the mask

class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)#   instantiated mae encode
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)#   instantiated mae decode class

    def forward(self, img):
        features, backward_indexes = self.encoder(img)# uses the forward function of the encoder class on the image to get the backward indexes and the features
        predicted_img, mask = self.decoder(features,  backward_indexes)#     uses the forward function of the decoder class to get the predicted image and mask
        return predicted_img, mask 

class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder : MAE_Encoder, num_classes=10) -> None:
        super().__init__()#  initializes super class
        #   takes the encoder class intializes its variables
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)#  applies linear transformation to the data
        #   with in features as the positional embedding and out feature as the number of classes

    def forward(self, img):
        patches = self.patchify(img)#   passes the image to patchify function in the encoder class and operates 2d convolution on it
        patches = rearrange(patches, 'b c h w -> (h w) b c')#   rearranges the image tensor 
        #   from (batch channel height width)
        #   to ((height * width) batch channel)
        patches = patches + self.pos_embedding #     does a element wise addtition to patches with positional embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)# expands the class tokens to adjust the patches and concatenates at the first dimension
        patches = rearrange(patches, 't b c -> b t c')#  rearranges the the tensor by swapping first and the second dimension 
        features = self.layer_norm(self.transformer(patches))#   apply layer normalization to the patches tensor that has been passed through the transformer
        features = rearrange(features, 'b t c -> t b c')#   rearranges the tensor by swapping the first and the second dimension
        logits = self.head(features[0])#     applys linear transformation to the first dimension of the features tensor
        return logits#  just returns the logits variable


if __name__ == '__main__':
    shuffle = PatchShuffle(0.75)#   initializes the patch shuffle class with a mask ratio of 75%
    a = torch.rand(16, 2, 10)#  this creates a numpy array with random numbers and of the size 16,2,10
    b, forward_indexes, backward_indexes = shuffle(a)#  then passes the random np array to the forward function of the patch shuffle class
    print(b.shape)

    img = torch.rand(2, 3, 32, 32)# this creates a numpy array with random numbers and of the size 3,2,32,32
    encoder = MAE_Encoder()#    instantiate encoder class
    decoder = MAE_Decoder()#    instantiate encoder class
    features, backward_indexes = encoder(img)#   passes the image to the encoder forward class to get the backward index and features
    print(forward_indexes.shape)
    predicted_img, mask = decoder(features, backward_indexes)#  passes the features and the backward indexes to get the predicted image and mask
    print(predicted_img.shape)
    loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)#    calculates the loss from the predicted image
    print(loss)
