import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_path ="../dataset"

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

print("Training images:", train_data.samples)
print("Validation images:", val_data.samples)
print("Classes:", train_data.class_indices)