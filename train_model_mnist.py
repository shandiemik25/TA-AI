import tensorflow as tf
from tensorflow.keras import layers, models
import keras_tuner as kt

# Load dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# --- Preprocessing --- 
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

batch_size = 32  # Lebih kecil, lebih ringan

ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_train = ds_train.map(preprocess).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
ds_test = ds_test.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# --- Model builder (lebih ringan) ---
def build_model(hp):
    model = models.Sequential()
    
    model.add(layers.Conv2D(
        filters=hp.Choice('conv1_filter', [16, 32, 64]),
        kernel_size=3,
        activation='relu',
        input_shape=(28, 28, 1)
    ))
    model.add(layers.MaxPooling2D())
    
    model.add(layers.Conv2D(
        filters=hp.Choice('conv2_filter', [16, 32]),
        kernel_size=3,
        activation='relu'
    ))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    
    model.add(layers.Dense(
        units=hp.Choice('dense_units', [32, 64]),
        activation='relu'
    ))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation='softmax'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice('lr', [1e-3, 5e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- Tuning --- 
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=5,
    factor=3,
    directory='kt_mnist_lite',
    project_name='mnist_lite'
)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

tuner.search(ds_train, validation_data=ds_test, epochs=5, callbacks=[stop_early])

best_model = tuner.get_best_models(1)[0]
best_model.save('best_mnist_model_lite.h5')
