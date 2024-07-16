import time
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, Rescaling

dataset_path = 'Animals'
batch_size = 32
img_height = 180
img_width = 180

train_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
val_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size 
)

#standardize data
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_train_ds = train_dataset.map(lambda x, y: (normalization_layer(x),y))
normalized_val_ds = val_dataset.map(lambda x, y: (normalization_layer(x),y))

#data augmentation
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
    RandomZoom(0.2)
])

#create Model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

#compiling model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

#Train Model
epochs = 10

start_time = time.time()

history = model.fit(
    normalized_train_ds,
    validation_data=normalized_val_ds,
    epochs=epochs,
    verbose=2
)

end_time = time.time()
total_time = end_time - start_time
print(f"Total training time:{total_time // 60} minutes and {total_time % 60} seconds")
      
#evaluate Model
loss, accuracy = model.evaluate(normalized_val_ds)
print(f"Validation Accuracy:{accuracy}")

#saving model
model.save('animal_classifier_model.keras')
