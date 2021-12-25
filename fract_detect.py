import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from utils import Config


config = Config()
fractured = os.path.join(config.options["data_dir"])
print(os.listdir(fractured))