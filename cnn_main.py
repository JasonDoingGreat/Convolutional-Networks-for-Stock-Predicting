import numpy as np
from keras.models import Sequential  
from keras.layers import Dense, Dropout, Activation, Flatten  
from keras.layers import Convolution2D, MaxPooling2D  
from keras.optimizers import SGD
from keras.utils import np_utils
from scipy import misc
import glob
import matplotlib.pyplot as plt
from PIL import Image
import math

seed = 7
np.random.seed(seed)
width = 1
height = 1


def r_squared(y_true, y_hat):
    ssr = 0
    sst = 0
    e = np.subtract(y_true, y_hat)
    y_mean = np.mean(y_true)
    for item in e:
        ssr += item**2
    for item in y_true:
        sst += (item - y_mean)**2
    r2 = 1 - ssr / sst
    return r2


def compile_model(model):
    lrate = 0.01
    sgd = SGD(lr=lrate, momentum=0.9, decay=1e-6, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd)
    return model


def create_model():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3,
                            border_mode='valid', 
                            input_shape=(100, 100, 3)))  
    model.add(Activation('relu'))  
    model.add(Convolution2D(32, 3, 3))  
    model.add(Activation('relu'))  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    model.add(Dropout(0.25))  
      
    model.add(Convolution2D(64, 3, 3, 
                            border_mode='valid'))  
    model.add(Activation('relu'))  
    model.add(Convolution2D(64, 3, 3))  
    model.add(Activation('relu'))  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    model.add(Dropout(0.25))  
      
    model.add(Flatten())  
    model.add(Dense(256))  
    model.add(Activation('relu'))  
    model.add(Dropout(0.5))

    model.add(Dense(2))  
    model.add(Activation('softmax'))  

    return model
    
    
def find_returns(data):
    returns = []
    for group in data:
        count = 30
        while count <= (len(group)-5):
            current_data = group[count-1]
            future_data = group[count+4]
            p1 = np.mean(current_data)
            p2 = np.mean(future_data)
            returns.append(math.log(p2/p1))
            count += 1
    return returns
    
    
def get_pixel_values():
    file_name = r'\figures_v2'
    pixels = []
    for filename in glob.glob(file_name + '\*.png'):
        im = misc.imread(filename)
        pixels.append(im)
    return pixels
    
    
def convert_image():
    file_name = r'\figures_v2'
    for filename in glob.glob(file_name + '\*.png'):
        img = Image.open(filename)
        img = img.convert('RGB')
        img.save(filename)
    
    
def plot_data(data):
    t = np.arange(0, 29, 1)
    file_name_number = 0
    fig = plt.figure(frameon=False, figsize=(width, height))
    for group in data:
        count = 30
        while count <= (len(group)-5):
            high = []
            low = []
            for item in group[count-30:count]:
                high.append(item[0])
                low.append(item[1])
            file_name = r'\fig_' + str(file_name_number)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.plot(t, high[0:-1], 'b', t, low[0:-1], 'g')
            fig.savefig(r'\figures_v2' + file_name, dpi=100)
            fig.clf()
            file_name_number += 1
            count += 1
    print('Created %d files!' % file_name_number)
    
    
def load_sample_data():
    original_data = extract_data()
    splitted_data = split_data(original_data)
    useful_data = extract_useful_data(splitted_data)
    return useful_data


def extract_useful_data(data):
    groups = []
    for group in data:
        temp_buffer = []
        for item in group:
            temp = [item[2], item[3]]
            temp = [float(i) for i in temp]
            temp_buffer.append(temp)
        groups.append(temp_buffer)
    return groups


def split_data(data):
    groups = []
    for item in data:
        temp_buffer = []
        for string in item:
            number = string.split(',')
            temp_buffer.append(number)
        groups.append(temp_buffer)
    return groups


def extract_data():
    file_name = r'\data.txt'
    infile = open(file_name, 'r')
    temp_buffer = []
    for line in infile:
        temp_buffer.append(line.strip('\n'))
    temp_buffer = temp_buffer[8:]
    i = 0
    groups = []
    temp = []
    for item in temp_buffer:
        if i != 390:
            temp.append(item)
            i += 1
        else:
            groups.append(temp)
            temp = []
            i = 0
    groups.append(temp)
    infile.close()
    return groups


def main():
    data = load_sample_data()
    plot_data(data)
    convert_image()
    x = np.asarray(get_pixel_values())
    y = np.asarray(find_returns(data))
    x_train = x[0:4340]
    y_train = y[0:4340]
    x_test = x[0:4340]
    y_test = y[0:4340]
#    y_true = y_test
#    y_train = np_utils.to_categorical(y_train, 2)
#    y_test = np_utils.to_categorical(y_test, 2)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0
    model = create_model()
    model = compile_model(model)

    # Fit the model
    epochs = 10
    model.fit(x_train, y_train, validation_data=(x_test, y_test), 
              nb_epoch=epochs,
              shuffle=True, batch_size=100, verbose=1)
#    scores = model.evaluate(x_test, y_test, verbose=0)
#    print('Accuracy: %.2f%%' % (scores[1] * 100))
    classes = model.predict_classes(x_test, verbose=0)
    classes = list(classes)
    y_test = list(y_test)
    r2 = r_squared(y_test, classes)
    print r2


if __name__ == '__main__':
    main()
