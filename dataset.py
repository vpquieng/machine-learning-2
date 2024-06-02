import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

img_height, img_width = 64, 64
batch_size = 20

# Loads the images from the datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    "MedBoxes/train",
    image_size=(img_height, img_width),
    batch_size=batch_size,
    validation_split=0.2,  # Split 20% of data for validation
    subset="training",
    seed=123
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "MedBoxes/train",
    image_size=(img_height, img_width),
    batch_size=batch_size,
    validation_split=0.2,  # Split 20% of data for validation
    subset="validation",
    seed=123
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "MedBoxes/test",
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = ["MedBox1", "MedBox2", "MedBox3", "MedBox4", "MedBox5"]
plt.figure(figsize=(20, 20))
for images, labels in train_ds.take(1):  # List of images and their labels
    for i in range(9):  # Displays 9 datasets in a 3x3 grid
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

# Preprocessing
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(5)  # Number of classes
])

model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Training
model.fit(
    train_ds,
    validation_data=val_ds,  # Evaluation of data while datasets are training
    epochs=10  # Times it goes over the datasets
)

# Evaluation
model.evaluate(test_ds)  # Loss and accuracy evaluation on the test dataset

# Plotting Predictions
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    classifications = model(images)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        index = np.argmax(classifications[i])  # Gets the highest number
        plt.title(f"Pred: {class_names[index]} | Real: {class_names[labels[i]]}")  # Prints out the output image of the highest number
        plt.axis("off")

plt.show()
