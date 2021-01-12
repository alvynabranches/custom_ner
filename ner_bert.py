import torch
from simpletransformers.classification import ClassificationModel

# pre-trained model
# model = ClassificationModel('roberta', 'roberta-base', use_cuda=torch.cuda.is_available())

# community model
model = ClassificationModel('bert', 'KB/bert-base-swedish-cased', use_cuda=torch.cuda.is_available())

# print(torch.cuda.is_available())
# print(torch.__version__)

# print(torch.cuda.device_count())

