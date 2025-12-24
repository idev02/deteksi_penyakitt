# =========================================
# TRAIN CNN – DATASET ASLI (FINAL & VALID)
# =========================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

# ================= PATH =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "train")
MODEL_DIR = os.path.join(BASE_DIR, "model")
IMG_DIR = os.path.join(BASE_DIR, "static", "images")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# ================= CONFIG =================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# ================= DATA =================
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

class_names = list(train_gen.class_indices.keys())
num_classes = len(class_names)

print("KELAS CNN:", class_names)

# ================= MODEL =================
base_model = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(base_model.input, output)
model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# ================= SAVE MODEL =================
model.save(os.path.join(MODEL_DIR, "hand_nail_disease_cnn.h5"))

# ================= EVALUATION =================
val_gen.reset()
y_score = model.predict(val_gen)
y_pred = np.argmax(y_score, axis=1)
y_true = val_gen.classes

# ================= CONFUSION MATRIX =================
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1)[:, None]

plt.figure(figsize=(8,6))
sns.heatmap(
    cm_norm,
    annot=cm,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – CNN (Validation Dataset)")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# ================= PRECISION–RECALL =================
y_true_bin = label_binarize(y_true, classes=range(num_classes))

plt.figure(figsize=(9,7))
for i, name in enumerate(class_names):
    precision, recall, _ = precision_recall_curve(
        y_true_bin[:, i],
        y_score[:, i]
    )
    ap = average_precision_score(
        y_true_bin[:, i],
        y_score[:, i]
    )
    plt.plot(recall, precision, label=f"{name} (AP={ap:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve – CNN (Validation Dataset)")
plt.legend(fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "precision_recall_curve.png"), dpi=150)
plt.close()

# ================= REPORT =================
df = pd.DataFrame(
    classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
).transpose()

df.to_csv(os.path.join(MODEL_DIR, "classification_report.csv"))

print("✅ SEMUA OUTPUT DARI DATASET VALIDASI BERHASIL DIBUAT")
