import tensorflow as tf
from tensorflow.keras. models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow. keras.layers import Dropout, Flatten
import matplotlib.pyplot as plt
import seaborn as sns

mnist = tf.keras. datasets .mnist
(x_train, y_train) , (x_test, y_test ) = mnist. load_data() # Data l oading
x_train, x_test = x_train/255.0 , x_test/255.0 #Normal izing the data

sns.heatmap(x_train[0])
plt.show()

model = Sequential([
Flatten(input_shape=(28, 28)),
Dense(128, activation="relu"),
Dropout (0,2),
Dense(10) ])

predictions = model(x_train[ :1]).numpy()
predictions

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile (optimizer="adam",loss = loss_fn, metrics=["accuracy"])

model.fit(x_train, y_train, epochs=5)

model. evaluate(x_test, y_test, verbose=2)

val = model.fit (x_train, y_train, epochs=5, validation_data=(x_test, y_test), batch_size=32)

plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("epoch")
plt.plot(val.history["accuracy"])
plt.plot (val.history[ "val_accuracy"])
plt.legend(["train","val"])
plt.show()

