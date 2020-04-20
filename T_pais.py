import numpy as np
from skimage import io, color, exposure, transform
from sklearn.model_selection import train_test_split
import os
import glob
import h5py
import sys

from keras.preprocessing.image import ImageDataGenerator
#from keras.models import Sequential, model_from_json
from keras.models import Sequential, model_from_json, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from keras.optimizers import SGD
#from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K
K.set_image_data_format('channels_first')

from matplotlib import pyplot as plt
#%matplotlib inline

from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
#from keras.models import Sequential
#from keras.layers import Dense
import time
import datetime 

print ("Ini=", datetime.datetime.now())
inicio=time.time()

IMG_SIZE = 48
nb_epoch = 30

csv = sys.argv[1]
NUM_CLASSES = int(sys.argv[2])
batch_size = int(sys.argv[3])

print("file csv =",csv)
print("batch_size =",batch_size,"  epoch =",nb_epoch, "  img_size=",IMG_SIZE,"class=",NUM_CLASSES)



def preprocess_img(img):
    # Histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img,-1)

    return img


def get_class(img_path):
    return int(img_path.split('/')[-2])

try:
    with  h5py.File('X.h5') as hf: 
        X, Y = hf['imgs'][:], hf['labels'][:]
    print("\nLoaded images from X.h5")
    
except (IOError,OSError, KeyError):  
    print("\nError in reading X.h5. Processing all images...")
    root_dir = 'Training/'
    imgs = []
    labels = []

    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
    np.random.shuffle(all_img_paths)
    for img_path in all_img_paths:
        try:
            img = preprocess_img(io.imread(img_path))
            label = get_class(img_path)
            imgs.append(img)
            labels.append(label)

            if len(imgs)%1000 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
        except (IOError, OSError):
            print('missed', img_path)
            pass

    X = np.array(imgs, dtype='float32')
    Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]
    #Y = np.eye(NUM_CLASSES, dtype='str')[labels]

    with h5py.File('X.h5','w') as hf:
        hf.create_dataset('imgs', data=X)
        hf.create_dataset('labels', data=Y)

def cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=( 3, IMG_SIZE, IMG_SIZE),
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model


#model = load_model('model_Alemao.h5')
model = cnn_model()
# let's train the model using SGD + momentum (how original).
lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
#optimizer=Adam()
model.compile(loss='categorical_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])


def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))




print ("\nEXECUCAO DO TREINAMENTO")   # TREINAMENTO----------------------
history = model.fit(X, Y,
          batch_size=batch_size,
          epochs=nb_epoch,
          validation_split=0.2,
#          validation_split=0.3,
          shuffle=True,
          verbose=0,
          #workers=4,
          callbacks=[
              LearningRateScheduler(lr_schedule),
                    ModelCheckpoint('model.h5',save_best_only=True)
             ]
            )
print ("\nFIM DA EXECUCAO DO TREINAMENTO")
print ("FIM1=", datetime.datetime.now())
fim1=time.time()
print ("fim1=", fim1-inicio)


'''
from keras.utils import plot_model
plot_model(model, to_file='model_Alemao.png')
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy(Alemao)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss(Alemao)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
'''


#------------------------SAVE MODEL and WEIGHTS----------------------------------
# serialize model to JSON
#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
#model.save_weights("model_weights.h5")
#print("Saved model to disk")
#-----------------------------------------------------------

                                      # TESTE----------------------
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import os
test = pd.read_csv(csv,sep=';')

X_test = []
y_test = []
i = 0

for file_name, class_id  in zip(list(test['Filename']), list(test['ClassId'])):
#for  class_id, file_name,  in zip(list(test['ClassId']), list(test['Filename'])):
    img_path = os.path.join('Testing/',file_name)
    X_test.append(preprocess_img(io.imread(img_path)))
    y_test.append(class_id)
    
    
X_test = np.array(X_test)
y_test = np.array(y_test)

# predict probabilities for test set
yhat_probs = model.predict(X_test, verbose=0)

# predict crisp classes for test set
yhat_classes = model.predict_classes(X_test, verbose=0)

y_pred = model.predict_classes(X_test)
acc = np.sum(y_pred==y_test)/np.size(y_pred)
print("\nTest accuracy = {}".format(acc))
print("\ny_pred=", y_pred)

#                  CALCULO DAS METRICAS 
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
#yhat_classes = yhat_classes[:, 0]
yhat_classes = y_pred
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes, average='macro')
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes, average='macro')
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes, average='macro')
print('F1 score: %f' % f1)

# MATRIZ CONFUSAO
print("\nMatriz Confusao confusion_matriz\n", confusion_matrix(y_test, y_pred))

y_actu = pd.Series(y_test, name='Atual')
y_pred = pd.Series(y_pred, name='Pred')
df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Atual'], colnames=['Pred'], margins=True)
print("\n Matriz Confusao crosstab com totais\n",df_confusion)

#df_confusion = pd.crosstab(y_actu, y_pred)
#print("\n Matriz Confusao crosstab\n",df_confusion)
                       

#----------------------

print("\nCRIACAO (20% 80% DO TREINAMENTO) E CALCULO DA PRECISAO ")
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
#X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3, random_state=42)


datagen = ImageDataGenerator(featurewise_center=False, 
                            featurewise_std_normalization=False, 
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10.,)

datagen.fit(X_train)

# Reinstallise models 

model = cnn_model()
# let's train the model using SGD + momentum (how original).
lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
#optimizer=Adam()
model.compile(loss='categorical_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])


def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))


print("batch_size =",batch_size,"  epoca =",nb_epoch)
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            steps_per_epoch=X_train.shape[0],
                            epochs=nb_epoch,
                            verbose=0,
                            validation_data=(X_val, Y_val),
                            callbacks=[LearningRateScheduler(lr_schedule),
                                       ModelCheckpoint('model.h5',save_best_only=True)]
                           )

print ("Fim2=", datetime.datetime.now())
fim2=time.time()
print ("fim2=", fim2-fim1)
'''
from keras.utils import plot_model
plot_model(model, to_file='model_aug_Alemao.png')
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy(Alemao)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss(Alemao)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
'''

# predict probabilities for test set
yhat_probs = model.predict(X_test)

# predict crisp classes for test set
yhat_classes = y_pred = model.predict_classes(X_test)

acc = np.sum(y_pred==y_test)/np.size(y_pred)
print("\nTest accuracy = {}".format(acc))

#CALCULO DAS METRICAS
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
#yhat_classes = yhat_classes[:, 0]
yhat_classes = y_pred
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes, average='macro')
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes, average='macro')
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes, average='macro')
print('F1 score: %f' % f1)



print("\nMatriz Confusao Aug confusion_matriz\n", confusion_matrix(y_test, y_pred))
y_actu = pd.Series(y_test, name='Atual')
y_pred = pd.Series(y_pred, name='Pred')
#df_confusion = pd.crosstab(y_actu, y_pred)
#print("\n Matriz Confusao crosstab\n",df_confusion)
df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Atual'], colnames=['Pred'], margins=True)
print("\n Matriz Confusao Aug crosstab com totais\n",df_confusion)

model.summary()

model.count_params()


'''
# Escolha uma imagem dos dados de teste
img = X_test[0]

# Adicione a imagem a batch onde é o único membro.
img = (np.expand_dims(img,0))
#print(img.shape)

#Agora faça uma previsão sobre a imagem:
predictions_single = model.predict(img)

#print (" \nCALCULO DA PRECISAO DE UMA IMAGEM DO TESTE")
#max=-1
#imax=0
#for i in range(NUM_CLASSES):
#    print(  i,predictions_single[0,i])
#    if predictions_single[0,i] > max:
#        max = predictions_single[0,i]
#        imax = i
#print("indice =",imax,"  max =",max)

 #CALCULO DA PRECISAO DE IMAGENS DA WEB---------------------
    
#Load and Output the Images
### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import os
images = []

#print(K.image_data_format())
input_shape=( 3, IMG_SIZE, IMG_SIZE)

                          

print (" \nCALCULO DA PRECISAO DE IMAGENS DA WEB")

# Read all image into the folder
for filename in os.listdir("from_web"):
    img = Image.open(os.path.join("from_web", filename))
    img = img.resize((48, 48))
    plt.imshow(img)
#    plt.show()
    
    img = preprocess_img(img)
    img = (np.expand_dims(img,0))
    predictions_single = model.predict(img)
    #print(predictions_single)
    
    max=-1
    imax=0
    for i in range(NUM_CLASSES):
#        print(  i,predictions_single[0,i])
        if predictions_single[0,i] > max:
            max = predictions_single[0,i]
            imax = i
    #print("indice,max",imax,max)
    print("indice =",imax,"  max =",max)
    #images.append(img)


print ("gabarito alemao = 13(Preferencial,25(Homens trabalhando), 14(stop),  11(cruzamento), 3(60km)  ")
'''
