import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model('cats_vs_dogs_model.h5')

img_path = 'data/test/dogs/dog_528.jpg'  # sinov rasmi
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

pred = model.predict(x)
print("Natija:", "🐶 It" if pred[0][0] > 0.5 else "🐱 Mushuk")
