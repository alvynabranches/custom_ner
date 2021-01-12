# !pip install simpletransfomers
# !pip install torch===1.7.1 torchvision===0.8.2 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from simpletransformers.classification import ClassificationModel

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

data = pd.read_csv('ner_dataset.csv', encoding='latin1')
data.fillna(method='ffill')

data['Sentence #'] = LabelEncoder().fit_transform(data['Sentence #'])

# pre-trained model
model = ClassificationModel('roberta', 'roberta-base', use_cuda=torch.cuda.is_available())

# community model
model = ClassificationModel('bert', 'KB/bert-base-english-cased', use_cuda=torch.cuda.is_available())

