

# Import the necessary Libraries
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.utils import to_categorical, plot_model 


## Loading the dataset and performing splitting

(train_x, train_y), (test_x, test_y) = mnist.load_data()


## Checking the sizes of train and test split

print("The size of train_x is: {}".format(train_x.shape))
print("The size of train_y is: {}".format(train_y.shape))

print("The size of test_x is: {}".format(test_x.shape))
print("The size of test_y is: {}".format(test_y.shape))


## Peforming reshaping operations- Converting into 4-D


train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)




## Normalizing the values of image- converting in between 0 and 1
train_x = train_x/255.0
test_x = test_x/255.0


## One-hot encoding the labels

train_y = to_categorical(train_y, num_classes=10)
test_y = to_categorical(test_y, num_classes=10)


## The Model
model = Sequential()

model.add(Conv2D(96, (5,5), strides=4, padding='valid', 
                 input_shape=(28,28,1), activation='relu'))

model.add(MaxPool2D((2,2), strides=2))

model.add(Conv2D(256, (3,3), padding='same', activation='relu'))

model.add(MaxPool2D((2,2), strides=1))

model.add(Conv2D(384, (1,1), padding='same', activation='relu'))

model.add(Conv2D(384, (1,1), padding='same', activation='relu'))

model.add(Conv2D(256, (1,1), padding='same', activation='relu'))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dense(512, activation='relu'))

model.add(Dense(10, activation='softmax'))
model.summary()


## Compiling the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


## Fitting the model
model.fit(train_x, train_y, batch_size=128, epochs=1, verbose=1, validation_data=(test_x, test_y))


## Finding the loss and accuracy of the model
score = model.evaluate(test_x, test_y)

print('Test Loss:', score[0])
print('Test accuracy:', score[1])
