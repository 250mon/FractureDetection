from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
from utils import Config


def main():
    config = Config()
    data_dir = config.options["data_dir"]

    train_dir = os.path.join(data_dir, "train")
    fractured_dir = os.path.join(train_dir, "FRACTURED")
    unfractured_dir = os.path.join(train_dir, "UNFRACTURED")

    fractured_img = os.listdir(fractured_dir)
    print(fractured_img[:5])

    batch_size = 54
    target_size = (100, 100)
    train_datagen = ImageDataGenerator(rescale=1/255)
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=target_size, batch_size=batch_size,
                                                        classes=["FRACTURED", "UNFRACTURED"], class_mode='binary')

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation="tanh", input_shape=(100, 100, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(32, (3, 3), activation="tanh"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="tanh"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="tanh"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.summary()

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    total_sample = train_generator.n
    num_epoch = 5
    model.fit_generator(train_generator, steps_per_epoch=int(total_sample/batch_size), epochs=num_epoch)


if __name__ == "__main__":
    main()