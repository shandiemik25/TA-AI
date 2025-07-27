import tensorflow as tf
from tensorflow.keras import layers, models
import keras_tuner as kt

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Tambah dimensi channel dan convert ke tf.data.Dataset
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# --- Preprocessing function ---
def preprocess(image, label):
    image = tf.image.grayscale_to_rgb(image)           # (28,28,1) -> (28,28,3)
    image = tf.image.resize(image, [96, 96])           # Resize to 96x96
    image = tf.cast(image, tf.float32) / 255.0         # Normalize
    return image, label

# --- TF Dataset pipeline (hemat memori) ---
batch_size = 64

ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_train = ds_train.map(preprocess).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
ds_test = ds_test.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# --- Hyperparameter tuning area ---
# REMARK: Di bagian ini terjadi proses hyperparameter tuning

def build_model(hp):
    model = models.Sequential()

    model.add(layers.Conv2D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('conv_1_kernel', values=[3, 5]),
        activation='relu',
        input_shape=(96, 96, 3)
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.Dropout(rate=hp.Float('dropout_1', 0.2, 0.5, step=0.1)))

    model.add(layers.Conv2D(
        filters=hp.Int('conv_2_filter', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('conv_2_kernel', values=[3, 5]),
        activation='relu'
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.Dropout(rate=hp.Float('dropout_2', 0.2, 0.5, step=0.1)))

    model.add(layers.Flatten())
    model.add(layers.Dense(
        units=hp.Int('dense_units', min_value=64, max_value=256, step=64),
        activation='relu'
    ))
    model.add(layers.Dropout(rate=hp.Float('dropout_3', 0.2, 0.5, step=0.1)))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# --- Tuner ---
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='kt_mnist',
    project_name='mnist_96x96'
)

# --- Early stopping ---
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# --- Search best hyperparameters ---
tuner.search(ds_train, validation_data=ds_test, epochs=10, callbacks=[stop_early])

# --- Get best model ---
best_model = tuner.get_best_models(num_models=1)[0]

# --- Save the best model ---
best_model.save('best_mnist_model_96x96.h5')
