##link to dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
import os
from glob import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sklearn
import skimage
from tqdm import tqdm
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import keras
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from keras.optimizers import SGD
from keras.layers import Dense, Flatten
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.utils.np_utils import to_categorical

train_dir = "..." #path to training images
test_dir =  "..." #path to testing images

## store the images in a numpy array
def get_data(folder):
    X = []
    y = []
    for fname in os.listdir(folder):
        if not fname.startswith('.'):
            if fname in ['NORMAL']:
                label = 0
            elif fname in ['PNEUMONIA']:
                label = 1
            else:
                label = 2
            for img_file_name in tqdm(os.listdir(folder + fname)):
                img_file = cv2.imread(folder + fname + '/' + img_file_name)
                if img_file is not None:
                    img_file = skimage.transform.resize(img_file, (150, 150, 3))
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y
X_train, y_train = get_data(train_dir)
X_test, y_test= get_data(test_dir)

##one hot encoding
y_trainHot = to_categorical(y_train, num_classes = 2)
y_testHot = to_categorical(y_test, num_classes = 2)

##compute class weight
map_characters1 = {0: 'No Pneumonia', 1: 'Yes Pneumonia'}
cw1 = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

##load pretrained weights
w_path1 = './vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
w_path2 = './inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

##load the pretrained models
pretrained_model_1 = VGG16(weights = w_path1, include_top=False, input_shape=(150, 150, 3))
pretrained_model_2 = InceptionV3(weights = w_path2, include_top=False, input_shape=(150, 150, 3))
opt1 = keras.optimizers.RMSprop(lr=0.0001)

def pretrainednet(xtrain,ytrain,xtest,ytest,pretrainedmodel,pretrainedweights,classweight,numclasses,numepochs,optimizer,labels):
    base = pretrained_model_1 
    x = base.output
    x = Flatten()(x)
    pred = Dense(numclasses, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=pred)
    for layer in base.layers:
        layer.trainable = False
    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer, 
                  metrics=['accuracy'])
    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
    model.summary()
    history = model.fit(xtrain,ytrain, epochs=numepochs, class_weight=classweight, validation_data=(xtest,ytest), verbose=1,callbacks = [MetricsCheckpoint('logs')])
    score = model.evaluate(xtest,ytest, verbose=0)
    print('\nKeras CNN - accuracy:', score[1], '\n')
    y_pred = model.predict(xtest)
    print('\n', sklearn.metrics.classification_report(np.where(ytest > 0)[1], np.argmax(y_pred, axis=1), target_names=list(labels.values())), sep='') 
    Y_pred_classes = np.argmax(y_pred,axis = 1) 
    Y_true = np.argmax(ytest,axis = 1) 
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
    plotKerasLearningCurve()
    plt.show()
    plot_learning_curve(history)
    plt.show()
    plot_confusion_matrix(confusion_mtx, classes = list(labels.values()))
    plt.show()
    return model
pretrainednet(X_train, y_trainHot, X_test, y_testHot,pretrained_model_1,w_path1,cw1,2,3,opt1,map_characters1)
pretrainednet(X_trainRosReshaped, Y_trainRosHot, X_testRosReshaped, Y_testRosHot,pretrained_model_2,w_path2,class_weight2,2,6,opt1,map_characters1)
