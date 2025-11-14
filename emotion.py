import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ✅ Ensure TensorFlow doesn't overload your CPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ✅ Paths
base_dir = os.path.join(os.getcwd(), "dataset")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "test")

# ✅ Verify dataset folder
if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    raise FileNotFoundError("❌ Dataset folders not found! Please make sure 'dataset/train' and 'dataset/test' exist.")

# ✅ Image Generators
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical",
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical",
)

# ✅ Define Model
model = Sequential([
    Input(shape=(48, 48, 1)),
    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(256, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(7, activation="softmax"),
])

# ✅ Compile Model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# ✅ Print model summary
model.summary()

# ✅ Train Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
)

# ✅ Save the model
model.save("fer2013_emotion_model.h5")
print("\n✅ Model saved as 'fer2013_emotion_model.h5'")

# ✅ Plot accuracy and loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
