import gensim
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Dense



filename = 'trmodel'
model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)



vector_dict = {}
for word in model.index_to_key: 
    vector_dict[word] = model.get_vector(word)


df = pd.DataFrame(vector_dict).T
print(df.head())

class AutoEncoders(Model):

    def __init__(self, output_units):
        super().__init__()
        activ_func = "LeakyReLU"
        
        self.encoder = Sequential(
            [   
                Dense(320, activation=activ_func),
                Dense(160, activation=activ_func),
                Dense(80, activation=activ_func),
                Dense(40, activation=activ_func),
            ]
        )

        self.decoder = Sequential(
            [
                Dense(40, activation=activ_func),
                Dense(80, activation=activ_func),
                Dense(160, activation=activ_func),
                Dense(320, activation=activ_func),
                Dense(output_units, activation="linear")
            ]
        )


    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

INPUT = list(vector_dict.values())
OUTPUT = list(vector_dict.keys())
X = np.asarray(INPUT)
y = np.asarray(OUTPUT)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1, shuffle = False)

auto_encoder = AutoEncoders(X.shape[1])
auto_encoder.compile(
    loss='mse',
    metrics=['accuracy'],
    optimizer='adam'
)

history = auto_encoder.fit(
    X_train,
    X_train,
    epochs=1,
    batch_size=64,
    validation_data=(X_test, X_test)
)
auto_encoder.summary()
auto_encoder.encoder.summary()
auto_encoder.decoder.summary()

history = auto_encoder.fit(
    X_train,
    X_train,
    epochs=100,
    batch_size=128,
    validation_data=(X_test, X_test)
)

encoded = auto_encoder.encoder(X).numpy()
decoded = auto_encoder.decoder(encoded).numpy()

print(encoded)

print(encoded.shape)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

model.vectors = model.vectors[:,0:40]
model.vector_size = 40

for word, vector in zip(list(model.key_to_index.values()), encoded):
    model.vectors[word] = vector

model.syn0norm = None
model.fill_norms(force=True)
model.get_normed_vectors()


vector_dict_new = {}
for word in model.index_to_key: 
    vector_dict_new[word] = model.get_vector(word)


model.save_word2vec_format("shrinked_model", binary=True)
