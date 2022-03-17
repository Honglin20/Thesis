from dataloader import Dataloader
from models.dgan import DGAN
import keras
from keras.layers import Dense
from models.pnn import Probability_CLF_Mul
import numpy as np
from utils import load_model

loader = Dataloader()
trained_gens = None

acgan = ACGAN()
# acgan.train(loader.get_dataset(0), 14000)

data, labels = loader.get_dataset(0)

gen, dis = load_model('saved_model/trained_acgans/', 0)
model = keras.models.Model(dis.input, dis.layers[-2].output)
model.compile(loss='binary_crossentropy', optimizer='adam')

features = dis.layers[-3].output
validity = Probability_CLF_Mul(1)(features)
label = dis.layers[-1](features)
model = keras.models.Model(dis.layers[1].input, [validity, label])
for layer in model.layers[:-2]:
    layer.trainable = False
model.layers[-1].trainable = False
model.compile(optimizer='adam', loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], metrics=['mse'],
              loss_weights=[100000, 0.00001])

data_idx = model.predict(data) > 0.5
data_idx = data_idx.reshape(-1, )
data = data[data_idx]
labels = labels[data_idx]

def gen_samples(model_gen, num_samples=2000):
    r, c = num_samples, 2
    noise = np.random.normal(0, 1, (r, 100))
    sampled_labels = np.random.randint(2, size=r)
    gen_imgs = model_gen.predict([noise, sampled_labels])
    return gen_imgs

data = np.concatenate([data, gen_imgs])
validation = np.concatenate([np.zeros_like(labels), np.ones_like(labels)])
labels = np.concatenate([labels, sampled_labels])
model.fit(data, [validation, labels], epochs=10)

from dataloader import Dataloader
loader = Dataloader()
for model in models['dis']:
    for i in range(5):
        print(np.mean(model.predict(loader.get_dataset(i, datatype='test')[0])[0]))
