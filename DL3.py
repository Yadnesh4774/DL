import tensorflow as tf
print(tf.__version__)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD
import numpy as np
import random
import matplotlib.pyplot as plt


import os
os.environ['CUDA _VISIBLE DEVICES'] ='-1'


x_train = np.loadtxt('input.csv',delimiter =',' , usecols=range(23316))
y_train = np.loadtxt ('labels.csv',delimiter =',' , usecols=range(23316))



x_test = np.loadtxt('input_test.csv', delimiter =',')
y_test = np.loadtxt('labels_test.csv', delimiter =',')



x_train = x_train.reshape (len (x_train), 100, 100, 3)
y_train = y_train.reshape (len(y_train), 1)



x_test = x_test . reshape(len(x_test),100, 100,3)
y_test = y_test. reshape (len(y_test), 1)



x_train = x_train/255.0
x_test=x_test/255.0



print("Shape of X_train:", x_train.shape)
print("Shape of Y_train:", y_train.shape)
print("Shape of X_train:", x_test.shape)
print("Shape of X_train:", y_test.shape)



idx = random.randint(0, len(x_train))
plt.imshow(x_train[idx, :])
plt.show()


from tensorflow.keras.layers import BatchNormalization


opt = SGD(momentum=0.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics = ['accuracy'])



model.fit(x_train,y_train,epochs=10,batch_size=32, validation_data=(x_test, y_test))



model.evaluate(x_test, y_test)


idx2 = random.randint(0, len(y_test))
plt.imshow(x_test [idx2, :])
plt.show()



y_pred=model.predict(x_test[idx2,:].reshape(1, 100,100, 3))
y_pred = y_pred>0.5
if(y_pred==0):
  pred='dog'
else:
  pred='cat'
print("Our model says it is a",pred)



score=model.evaluate(x_test, y_test, verbose=0)
print ("Test Score: ", score[0])
print("Test accuracy: ", score[1])


model.summary ()


val = model.fit(x_train,y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))



plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel('epoch')
plt.plot(val.history['accuracy'])
plt.plot(val.history['val_accuracy'])
plt.legend(['train', 'val'])
plt.show()


