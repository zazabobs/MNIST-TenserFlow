import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Veri setini yükle
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Normalize et (0-255 arası değerleri 0-1 arası yap)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Modeli tanımla
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),        # 28x28 resmi düzleştir
    layers.Dense(128, activation='relu'),        # Gizli katman
    layers.Dropout(0.2),                         # Overfitting'i azalt
    layers.Dense(10, activation='softmax')       # Çıkış katmanı (0-9 rakamları)
])

# 4. Modeli derle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Eğit
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 6. Test et
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test doğruluk oranı: {test_acc:.4f}")

# 7. Eğitim grafiğini çiz
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Test Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()
