import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from tqdm import tqdm  # For progress bars
import matplotlib.pyplot as plt

Corpus = '''
elma suyu 
portakal suyu
vişne suyu
kayısı suyu

facebook şirketi
google şirketi
microsoft şirketi
mcdonalds şirketi
yahoo şirketi
'''

cumleler = [cumle for cumle in Corpus.split('\n') if cumle !='']
print(cumleler)

veriseti = [cumle.split() for cumle in cumleler]
veriseti = pd.DataFrame(veriseti, columns = ['girdi', 'cikti'])
print(veriseti)

kelimeler = list(veriseti.girdi) + list(veriseti.cikti)
tekil_kelimeler = set(kelimeler)
print(tekil_kelimeler)

pozitif = veriseti.copy()
for i in range(10):
    ekle = veriseti.copy()
    pozitif = pd.concat((pozitif, ekle))
print(pozitif)

veri = pozitif

id2tok = dict(enumerate(tekil_kelimeler))
tok2id = {token: id for id, token in id2tok.items()}

print(id2tok)

X = veri[['girdi', 'cikti']].copy()
X.girdi = X.girdi.map(tok2id)
X.cikti = X.cikti.map(tok2id)
X

print(X)

# defining the Dataset class
class data_set(Dataset):
    def __init__(self, veri):
        X = veri[['girdi', 'cikti']].copy()
        
        X.girdi = X.girdi.map(tok2id)
        X.cikti = X.cikti.map(tok2id)
        
        self.X = X.girdi.values
        self.y = X.cikti.values

    def __len__(self):
        return len(self.X)
  
    def __getitem__(self, index):
        return self.X[index], self.y[index]
  
  
dataset = data_set(veri)

# implementing dataloader on the dataset and printing per batch
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
for i, (batch_X, batch_y) in enumerate(dataloader):
    print(i, batch_X, batch_y)
    if i >= 2: break



class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.expand = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, input):
        # Encode input to lower-dimensional representation
        hidden = self.embed(input)
        # Expand hidden layer to predictions
        logits = self.expand(hidden)
        return logits
    
# Instantiate 2 model
EMBED_SIZE = 2 # Quite small, just for the tutorial
n_v = len(tekil_kelimeler)

model = Word2Vec(n_v, EMBED_SIZE)

# Relevant if you have a GPU:
device = torch.device('cpu')
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define training parameters
LR = 3e-4
EPOCHS = 1000
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


progress_bar = tqdm(range(EPOCHS * len(dataloader)))
running_loss = []
for epoch in range(EPOCHS):
    epoch_loss = 0
    for i, (batch_X, batch_y) in enumerate(dataloader):
        center, context = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(input=context)
        loss = loss_fn(logits, center)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        progress_bar.update(1)
    epoch_loss /= len(dataloader)
    running_loss.append(epoch_loss)



plt.plot(running_loss)
plt.show()

wordvecs =  model.expand.weight.cpu().detach().numpy()
#print(wordvecs)


def plotit(wordvecs):
    plt.plot(wordvecs[:,0], wordvecs[:,1], 'o')

    for i, cord in enumerate(wordvecs):
        plt.text(cord[0], cord[1], id2tok[i], fontsize=12)
    plt.show()

plotit(wordvecs)

"""
wordvecs_input =  model.embed.weight.cpu().detach().numpy()
plotit(wordvecs_input)

wordvecs_input =  model.expand.weight.cpu().detach().numpy()
plotit(wordvecs_input)

plotit(wordvecs_input + wordvecs)
"""


#  TODO: ----------SORU1----------
#  Size verilen 11 kelimelik basit word2vec Yeni Kelime/Kelime
#  Vektörü Ekleme. Bunun için eğitilmiş word2vec kodunu yeniden
#  eğitmeden, yeni kelime ekleyeceksiniz. Örneğin muz kelimesi
#  eklenecekse var olan meyve kelimelerinin vektörlerinin ortalaması
#  alınacak. 

print("SORU 1 IN CEVABI")

import numpy as np

# Step 1: Calculate the average vector of existing word vectors for the category (e.g., fruits)
fruit_vectors = []
fruit_words = ['elma', 'portakal', 'vişne', 'kayısı']
for word in fruit_words:
    word_id = tok2id[word]
    word_vector = wordvecs[word_id]
    fruit_vectors.append(word_vector)
fruit_average_vector = np.mean(fruit_vectors, axis=0)

# Step 2: Add the new word "muz" to the vocabulary with the calculated average vector
new_word = 'muz'
new_word_id = len(id2tok)
id2tok[new_word_id] = new_word
tok2id[new_word] = new_word_id
wordvecs = np.vstack((wordvecs, fruit_average_vector))

# Step 3: Update the expand layer weights of the Word2Vec model with the new vocabulary and vectors
model.expand = nn.Linear(EMBED_SIZE, n_v, bias=False)
model.expand.weight = nn.Parameter(torch.Tensor(wordvecs), requires_grad=True)

# Step 4: Update the embed layer weights of the Word2Vec model with the new vocabulary and vectors
model.embed = nn.Embedding(len(id2tok), EMBED_SIZE)
model.embed.weight = nn.Parameter(torch.Tensor(wordvecs), requires_grad=True)

# Plot the updated word vectors

wordvecs_output =  model.embed.weight.cpu().detach().numpy()
plotit(wordvecs_output)

wordvecs_output2 = model.expand.weight.cpu().detach().numpy()
plotit(wordvecs_output2)

plotit(wordvecs_output + wordvecs_output2)