import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---- SETTINGS ---- #
IMG_SIZE = 224
BATCH_SIZE = 32
DATASET_PATH = r"C:\Users\KIIT0001\Desktop\blood_cancer_gui\model\dataset"
MODEL_PATH = r"C:\Users\KIIT0001\Desktop\blood_cancer_gui\model\blood_cancer_model.keras"

# ---- LOAD MODEL ---- #
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# ---- LOAD VALIDATION DATA ---- #
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

valid_classes = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith("__")
])
print("Classes found:", valid_classes)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    classes=valid_classes
)

# ---- PREDICT ---- #
print("Running predictions...")
y_pred_probs = model.predict(val_data, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_data.classes

# ---- CONFUSION MATRIX ---- #
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(valid_classes))))
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=valid_classes, zero_division=0))

# ---- PLOT ---- #
fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor('white')

row_totals = cm.sum(axis=1, keepdims=True)
cm_pct = np.nan_to_num(cm.astype(float) / row_totals * 100, nan=0.0)

heatmap = ax.imshow(cm, cmap='Blues', interpolation='nearest', aspect='auto')
for i in range(len(valid_classes)):
    for j in range(len(valid_classes)):
        count = cm[i, j]
        pct = cm_pct[i, j]
        text_color = 'white' if count > cm.max() / 2 else 'black'
        ax.text(j, i, f'{count}\n{pct:.1f}%',
                ha='center', va='center', color=text_color, fontsize=12)

ax.set_xticks(np.arange(len(valid_classes)))
ax.set_yticks(np.arange(len(valid_classes)))
ax.set_xticklabels(valid_classes, fontsize=11, fontweight='medium', rotation=45, ha='right')
ax.set_yticklabels(valid_classes, fontsize=11, fontweight='medium')
ax.set_xlabel('Predicted Class', fontsize=13, fontweight='bold', labelpad=12)
ax.set_ylabel('Actual Class', fontsize=13, fontweight='bold', labelpad=12)
ax.set_title('Confusion Matrix — Blood Cancer CNN Classifier\n(MobileNetV2, Transfer Learning)',
             fontsize=13, fontweight='bold', pad=16)

cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.set_ylabel('Count', rotation=-90, va='bottom', fontsize=11)

ax.set_ylim(len(valid_classes) - 0.5, -0.5)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)

accuracy = np.trace(cm) / cm.sum() * 100 if cm.sum() > 0 else 0.0
fig.text(0.5, 0.01,
         f'Overall Accuracy: {accuracy:.2f}%  |  Total Samples: {cm.sum()}',
         ha='center', fontsize=11, color='#333333', style='italic')

plt.tight_layout(rect=[0, 0.04, 1, 1])
output_path = r"C:\Users\KIIT0001\Desktop\blood_cancer_gui\model\confusion_matrix_real.png"
plt.savefig(output_path, dpi=180, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {output_path}")
