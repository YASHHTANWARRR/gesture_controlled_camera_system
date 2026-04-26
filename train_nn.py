import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import joblib

# Load dataset
df = pd.read_csv("gestures.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1]

# Encode labels
labels = y.unique()
label_map = {label:i for i,label in enumerate(labels)}
y_encoded = np.array([label_map[label] for label in y])
y_cat = to_categorical(y_encoded)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2)

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dense(64, activation='relu'),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=20, batch_size=16)

# Save
model.save("gesture_nn.keras")
joblib.dump(label_map, "labels.pkl")

print("Model trained and saved!")