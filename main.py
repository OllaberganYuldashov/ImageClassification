# main.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# === Parametrlar ===
img_size = (150, 150)
batch_size = 32
epochs = 10

train_dir = 'data/train'
test_dir = 'data/test'

# === Ma'lumotlarni tayyorlash ===
train_datagen = ImageDataGenerator(
    rescale=1./255,            # piksel qiymatlarini [0,1] ga o‘tkazamiz
    rotation_range=15,         # rasmni biroz aylantirish
    width_shift_range=0.1,     # gorizontal siljitish
    height_shift_range=0.1,    # vertikal siljitish
    shear_range=0.1,           # kesish (shear)
    zoom_range=0.1,            # biroz kattalashtirish
    horizontal_flip=True,      # chap-o‘ngni almashtirish
)

test_datagen = ImageDataGenerator(rescale=1./255)  # test ma’lumotda faqat normallashtirish

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'  # faqat 2 sinf — cat va dog
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

# === CNN Model ===
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # binary classification (cat/dog)
])

# === Modelni kompilyatsiya qilish ===
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# === Model arxitekturasini ko‘rish ===
model.summary()

# === Modelni o‘qitish ===
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator
)

# === Natijani baholash ===
loss, acc = model.evaluate(test_generator)
print(f"\nTest aniqligi: {acc*100:.2f}%")

# === Modelni saqlash ===
model.save('cats_vs_dogs_model.h5')
print("✅ Model saqlandi: cats_vs_dogs_model.h5")
