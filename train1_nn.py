import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

# ---------------- LOAD DATA ----------------

print("📂 Loading CSV dataset...")

df = pd.read_csv("gestures.csv")

# ---------------- SPLIT FEATURES & LABEL ----------------

X = df.iloc[:, :-1]   # features
y = df.iloc[:, -1]    # labels (KEEP AS STRING)

# 🔥 CLEAN ONLY FEATURES (NOT LABELS)
X = X.apply(pd.to_numeric, errors='coerce')
X = X.dropna()

# match labels after cleaning
y = y.loc[X.index]

# convert to numpy
X = X.values.astype(np.float32)
y = y.values

print("✅ Dataset shape:", X.shape)
print("✅ Data type:", X.dtype)

# ---------------- ENCODE LABELS ----------------

labels = np.unique(y)
label_map = {label: i for i, label in enumerate(labels)}

y_encoded = np.array([label_map[label] for label in y])
y_cat = to_categorical(y_encoded)

print("✅ Labels:", label_map)

# ---------------- TRAIN TEST SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------

print("🧠 Building model...")

model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(len(labels), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------- TRAIN ----------------

print("🚀 Starting training...")

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

# ---------------- EVALUATE ----------------

loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {acc:.4f}")

# ---------------- SAVE ----------------

print("💾 Saving model...")

model.save("gesture_nn.keras")
joblib.dump(label_map, "labels.pkl")

print("🎉 Model trained and saved successfully!")