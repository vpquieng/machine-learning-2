import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

img_height, img_width = 64, 64
batch_size = 20

# Directory paths
train_dir = "MedBoxes/train"
test_dir = "MedBoxes/test"
dummy_dir = "MedBoxes/dummyPics"

# Ensure the directories are correctly structured
if not os.path.exists(train_dir):
    raise ValueError(f"Training directory '{train_dir}' does not exist.")
if not os.path.exists(test_dir):
    raise ValueError(f"Testing directory '{test_dir}' does not exist.")
if not os.path.exists(dummy_dir):
    raise ValueError(f"Dummy directory '{dummy_dir}' does not exist.")
if not any(os.scandir(dummy_dir)):
    raise ValueError(f"No images found in directory '{dummy_dir}'. Ensure the directory has images in subdirectories.")

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=123
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=123
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

dummy_ds = tf.keras.utils.image_dataset_from_directory(
    dummy_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Add unknown class
class_names = ["MedBox1", "MedBox2", "MedBox3", "MedBox4", "MedBox5", "Unknown"]

# Augment data
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2)
])

# Preprocessing and creating the model
model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(class_names))  # Number of classes including "Unknown"
])

model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Training the model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20  
)

# Evaluating the model
model.evaluate(test_ds)

# Plotting predictions with rejection
plt.figure(figsize=(10, 10))
for images, labels in dummy_ds.take(1):
    classifications = model(images)
    for i in range(min(9, len(images))):  # Ensure we don't access out-of-bounds indices
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        index = np.argmax(classifications[i])
        confidence = tf.nn.softmax(classifications[i])[index]
        if confidence < 0.8:  # Adjusted threshold for rejection
            pred_label = "Rejected"
        else:
            pred_label = class_names[index]
        plt.title(f"Pred: {pred_label} | Conf: {confidence:.2f}")
        plt.axis("off")

plt.show()
