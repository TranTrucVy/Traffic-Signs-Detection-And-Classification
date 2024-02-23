from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

data = []
labels = []
classes = 19

cur_path = os.getcwd()
for i in range(classes):
    path = os.path.join(cur_path, 'TrainData', str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image_path = os.path.join(path, a)
            image = Image.open(image_path).convert('RGB')  # Keep the image in color
            image = image.resize((30, 30))
            image = np.array(image)

            data.append(image)
            labels.append(i)
        except Exception as e:
            print('Error loading image:', image_path, '\nError:', e)

data = np.array(data, dtype='float32') / 255.0
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

model = Sequential([
    Conv2D(32, (5, 5), activation='relu', input_shape=(30, 30, 3)),
    Conv2D(32, (5, 5), activation='relu'),
    MaxPool2D(2, 2),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D(2, 2),
    Dropout(0.25),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(19, activation='softmax')  
])


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=64, epochs=15, validation_data=(X_test, y_test))

model.save('model.h5')