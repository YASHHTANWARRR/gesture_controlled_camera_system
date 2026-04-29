import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
import os

IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10

DATASET_PATH = "dataset/"

# ---------------- DEBUG CHECKS ----------------

print("🔍 Checking dataset...")

if not os.path.exists(DATASET_PATH):
    raise Exception("❌ dataset/ folder not found!")

classes = os.listdir(DATASET_PATH)
print("Classes found:", classes)

if len(classes) == 0:
    raise Exception("❌ No class folders inside dataset/")

# ---------------- LOAD DATASET ----------------

print("📂 Loading dataset...")

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=123
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=123
)

class_names = train_ds.class_names
print("✅ Classes:", class_names)

# ---------------- PERFORMANCE OPT ----------------

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ---------------- MODEL ----------------

print("🧠 Building model...")

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(64,64,3)),

    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),   # 🔥 prevents overfitting
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------- TRAIN ----------------

print("🚀 Starting training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1
)

# ---------------- SAVE ----------------

print("💾 Saving model...")

model.save("gesture_new.keras")

label_map = {name: i for i, name in enumerate(class_names)}
joblib.dump(label_map, "labels.pkl")

print("✅ Training complete and model saved!")