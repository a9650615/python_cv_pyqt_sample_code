
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model

BATCH_SIZE = 128
SAVED_MODEL_DIR = './saved_model'

mnist = tf.keras.datasets.mnist
(images_train, labels_train),(images_test, labels_test) = mnist.load_data()

(ds_train_data, ds_val_data), info = tfds.load(
    name='mnist',
    split=['train', 'test'],
    with_info=True,
    as_supervised=True,
)

num_classes = info.features['label'].num_classes

def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE

ds_train = (
    ds_train_data
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(info.splits['train'].num_examples)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

ds_val = (
    ds_val_data
    .map(preprocess, AUTOTUNE)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTOTUNE)
)

inputs = layers.Input(shape=(28, 28, 1), name='input')

x = layers.Conv2D(24, kernel_size=(6, 6), strides=1)(inputs)
x = layers.BatchNormalization(scale=False, beta_initializer=Constant(0.01))(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(rate=0.25)(x)

x = layers.Conv2D(48, kernel_size=(5, 5), strides=2)(x)
x = layers.BatchNormalization(scale=False, beta_initializer=Constant(0.01))(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(rate=0.25)(x)

x = layers.Conv2D(64, kernel_size=(4, 4), strides=2)(x)
x = layers.BatchNormalization(scale=False, beta_initializer=Constant(0.01))(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(rate=0.25)(x)

x = layers.Flatten()(x)
x = layers.Dense(200)(x)
x = layers.BatchNormalization(scale=False, beta_initializer=Constant(0.01))(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(rate=0.25)(x)

predications = layers.Dense(num_classes, activation='softmax', name='output')(x)

model = Model(inputs=inputs, outputs=predications)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(images_train, labels_train, epochs=10)

model.summary()

tf.saved_model.save(model, SAVED_MODEL_DIR)

test_loss, test_acc = model.evaluate(images_test, labels_test)
print('Test accuracy:', test_acc)

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

print("Convert model")

with open('mnist.tflite', 'wb') as f:
    f.write(tflite_model)
    