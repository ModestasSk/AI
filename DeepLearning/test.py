import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
if tf.keras.models.load_model('number_reader.model')==False:
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3)

    val_loss, val_acc = model.evaluate(x_test, y_test)

    print(val_loss, val_acc)

    model.save('number_reader.model')
    new_model = tf.keras.models.load_model('number_reader.model')
    predictions = new_model.predict(x_test)

    print(np.argmax(predictions[1042]))
    plt.imshow(x_test[1042])
    plt.show()
else:
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    new_model = tf.keras.models.load_model('number_reader.model')
    predictions = new_model.predict(x_test)

    print(np.argmax(predictions[1090]))
    plt.imshow(x_test[1090])
    plt.show()