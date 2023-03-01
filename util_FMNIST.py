import tensorflow as tf
import tensorflow_datasets as tfds


def preprocess_data(image, label):
    # Normalize pixel values to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0

    # One-hot encode labels
    label = tf.one_hot(label, depth=10)
    label = tf.cast(label, tf.int32)


    return image, label

def load_data():
    # Load Fashion-MNIST data
    data, info = tfds.load('fashion_mnist', split=['train[:80%]', 'train[80%:]', 'test'], with_info=True, as_supervised=True)
    train_data, val_data, test_data = data[0], data[1], data[2]
    
    
    # Get the number of classes
    num_classes = info.features['label'].num_classes


    return train_data, val_data, test_data, num_classes