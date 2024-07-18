import torch

debug = False 
image_path = "/data/clipdata/Images"
#image_path = "/media/student/isaacsim/clipdata/Images"
#image_path = "C:/Moein/AI/Datasets/Flicker-8k/Images"
captions_path = "/data/clipdata"
#captions_path = "/media/student/isaacsim/clipdata"
#captions_path = "C:/Moein/AI/Datasets/Flicker-8k"
batch_size = 8 
num_workers = 4
lr = 1e-3
head_lr = 1e-3
image_encoder_lr = 1e-3
text_encoder_lr = 1e-4
weight_decay = 1e-3
patience = 2
factor = 0.5
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'resnet50'
image_embedding = 2048
text_encoder_model = "distilbert-base-uncased"
text_embedding = 768
text_tokenizer = "distilbert-base-uncased"
max_length = 200

pretrained = True
trainable = True # for both image encoder and text encoder
temperature = 1.0

# image size
size = 224

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256 
dropout = 0.1
