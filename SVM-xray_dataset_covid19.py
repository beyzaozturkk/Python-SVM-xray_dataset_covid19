import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Klasör yolları
train_dir = 'xray_dataset_covid19/train'
test_dir = 'xray_dataset_covid19/test'

# Görsellerin yükleneceği listeler
train_images = []
train_labels = []
test_images = []
test_labels = []

# Görselleri yükleme fonksiyonu
def load_images_from_directory(directory, label, images_list, labels_list):
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Görsellerin uzantılarını kontrol et
                # Dosya yolunu oluştur
                file_path = os.path.join(dirname, filename)
                
                # Görseli oku
                img = cv2.imread(file_path)
                
                if img is None:
                    print(f"Warning: Unable to read image {file_path}")
                    continue
                
                # Görselleri 64x64 boyutuna küçültme
                img = cv2.resize(img, (64, 64))  
                
                # Piksel değerlerini [0, 1] aralığına normalize etme
                img = img / 255.0  
                
                # Görseli ve etiketini listeye ekle
                images_list.append(img)
                labels_list.append(label)

# Train ve Test verilerini yükle
# Train verilerini yükleyelim
load_images_from_directory(os.path.join(train_dir, 'NORMAL'), 0, train_images, train_labels)  # Normal etiketini 0 olarak kabul ediyoruz
load_images_from_directory(os.path.join(train_dir, 'PNEUMONIA'), 1, train_images, train_labels)  # Pneumonia etiketini 1 olarak kabul ediyoruz

# Test verilerini yükleyelim
load_images_from_directory(os.path.join(test_dir, 'NORMAL'), 0, test_images, test_labels)
load_images_from_directory(os.path.join(test_dir, 'PNEUMONIA'), 1, test_images, test_labels)

# Verilerin doğru şekilde yüklendiğinden emin olmak için şekillerini yazdıralım
print(f"Train images shape: {np.array(train_images).shape}")
print(f"Train labels shape: {np.array(train_labels).shape}")
print(f"Test images shape: {np.array(test_images).shape}")
print(f"Test labels shape: {np.array(test_labels).shape}")



import random

# Rastgele 10 görseli seç
random_indices = random.sample(range(len(train_images)), 10)

# 10 görseli yazdırmak için alt grafikler oluştur
plt.figure(figsize=(15, 8))

for i, idx in enumerate(random_indices, 1):
    plt.subplot(2, 5, i)  # 2 satır ve 5 sütunlu bir düzen
    plt.imshow(train_images[idx])  # Görseli göster
    plt.title(f"Label: {'Pneumonia' if train_labels[idx] == 1 else 'Normal'}")  # Etiketleri yazdır
    plt.axis('off')  # Eksenleri kapat

plt.tight_layout()
plt.show()



# Görselleri ve etiketleri numpy array'e dönüştür
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Görselleri tek boyutlu vektörlere dönüştür
train_images_flattened = train_images.reshape(train_images.shape[0], -1)
test_images_flattened = test_images.reshape(test_images.shape[0], -1)

# Etiketleri sayısal hale getirelim (0 ve 1 zaten sayısal ama SVM'de işlem kolaylığı açısından)
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Eğitim ve test verilerini ayırma
X_train, X_val, y_train, y_val = train_test_split(train_images_flattened, train_labels_encoded, test_size=0.2, random_state=42)

# SVM modelini oluşturma ve eğitme
svm_model = SVC(kernel='linear', random_state=42)  # Lineer kernel kullanıyoruz
svm_model.fit(X_train, y_train)

# Doğrulama setinde modelin başarımını kontrol etme
y_val_pred = svm_model.predict(X_val)
print("Validation accuracy: ", accuracy_score(y_val, y_val_pred))
print("Classification report:\n", classification_report(y_val, y_val_pred))

# Test verisi ile modelin doğruluğunu ölçme
y_test_pred = svm_model.predict(test_images_flattened)
print("\nTest accuracy: ", accuracy_score(test_labels_encoded, y_test_pred))
print("Test Classification report:\n", classification_report(test_labels_encoded, y_test_pred))



# Modelin tahminlerini al
y_pred = svm_model.predict(test_images_flattened)

# Rastgele 10 görsel seç
random_indices = random.sample(range(len(test_images)), 20)

# 10 görseli yazdırmak için alt grafikler oluştur
plt.figure(figsize=(15, 8))

for i, idx in enumerate(random_indices, 1):
    plt.subplot(4, 5, i)  # 2 satır ve 5 sütunlu bir düzen
    plt.imshow(test_images[idx])  # Görseli göster
    
    # Gerçek etiket ve tahmin edilen etiket
    actual_label = 'Pneumonia' if test_labels[idx] == 1 else 'Normal'
    predicted_label = 'Pneumonia' if y_pred[idx] == 1 else 'Normal'
    
    # Başlıkta gerçek ve tahmin edilen etiketleri göster
    plt.title(f"Actual: {actual_label}\nPredicted: {predicted_label}", fontsize=10)
    plt.axis('off')  # Eksenleri kapat

plt.tight_layout()
plt.show()


