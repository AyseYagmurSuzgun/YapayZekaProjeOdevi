# 🧠 Alzheimer ve Parkinson Hastalıklarının EEG Verilerinin Yapay Zeka Analiziyle Erken Teşhisi ve Mobil Uygulama Takibi

## 📌 Proje Tanımı

Bu projede EEG (Elektroensefalografi) sinyalleri ve klinik veriler kullanılarak Alzheimer, Parkinson ve sağlıklı bireylerin sınıflandırılması amacıyla bir yapay zeka modeli geliştirilmiştir.

Geliştirilen model, beyin sinyallerinden elde edilen frekans bant güçlerini (delta, teta, alfa, beta, gama) ve klinik ölçümleri (MMSE, MoCA, UPDRS, yaş, cinsiyet) birlikte kullanarak multimodal bir karar destek sistemi oluşturur.

Ayrıca model çıktıları, kullanıcı dostu bir mobil uygulama arayüzü üzerinden görselleştirilerek doktorlar ve araştırmacılar için takip edilebilir hale getirilmiştir.

---

## 🎯 Projenin Amacı

Bu çalışmanın amacı, invaziv olmayan EEG sinyalleri ve klinik veriler kullanılarak nörodejeneratif hastalıkların erken teşhisinde kullanılabilecek yapay zekâ tabanlı bir karar destek sistemi geliştirmektir.

Ek olarak, sonuçların mobil ortamda erişilebilir olmasıyla klinik kullanım kolaylığı sağlanması hedeflenmiştir.

---

## 📊 Veri Seti

- Alzheimer veri seti: ds004504 (OpenNeuro / Kaggle)  
- Parkinson veri seti: ds004584 (OpenNeuro / Kaggle)  
- Toplam katılımcı sayısı: ~237 birey  

### 📌 Önemli Not
Veri setleri 2 saniyelik epoch segmentasyonu ile işlendiği için toplam örnek sayısı **8.113 EEG epoch** olmuştur.

---

## 🧪 EEG Kanalları

Veri setleri farklı EEG sistemleri (10-20 ve 10-10) kullandığı için kanal uyumsuzluğu bulunmaktadır. Bu nedenle ortak kanallar belirlenmiş ve veri setleri 14 kanal üzerinde standartlaştırılmıştır:

**Kullanılan EEG Kanalları:**

P4, P3, O2, O1, Fz, Fp2, Fp1, F8, F7, F4, F3, Cz, C4, C3

---

## 🧪 Sınıflar

- 0 → Healthy (Sağlıklı)
- 1 → Alzheimer
- 2 → Parkinson

---

## ⚙️ Veri Ön İşleme

Aşağıdaki ön işleme adımları uygulanmıştır:

- 1–40 Hz band-pass filtreleme  
- Ortalama referanslama  
- ICA ile artefakt temizleme  
- 2 saniyelik epoch segmentasyonu  
- Z-score normalizasyonu  
- Kanal standardizasyonu  

---

## 🧠 Özellik Çıkarımı

### EEG Özellikleri:
- Delta bant gücü  
- Teta bant gücü  
- Alfa bant gücü  
- Beta bant gücü  
- Gama bant gücü  

### Klinik Özellikler:
- MMSE  
- MoCA  
- UPDRS  
- Yaş  
- Cinsiyet  

Tüm özellikler birleştirilerek multimodal giriş vektörü oluşturulmuştur.

---

## 🏗️ Model Mimarisi

### EEG Modülü
- 1D CNN (özellik çıkarımı)
- Attention mekanizması

### Klinik Modülü
- Multi-Layer Perceptron (MLP)

### Fusion Katmanı
- EEG + klinik özelliklerin birleştirilmesi
- 3 sınıflı çıktı katmanı

---

## 🧪 Eğitim Detayları

- Framework: PyTorch  
- Epoch: 25  
- Optimizer: Adam  
- Loss Function: Focal Loss  
- Class imbalance: Class Weight  

---

## 📊 Değerlendirme Metrikleri

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

---

## 📈 Sonuçlar

- Accuracy: ~%63  
- Macro F1-score: ~0.63  

### Gözlemler
- Parkinson sınıfında yüksek recall değeri elde edilmiştir  
- Alzheimer sınıfında yüksek precision gözlemlenmiştir  
- Healthy sınıfında sınıflar arası benzerlik nedeniyle performans düşmüştür  

---

## 📉 Deneyler

Farklı veri bölme oranları test edilmiştir:

- 80/20  
- 70/15/15  
- 60/20/20  
- 50/25/25  

**Sonuç:** Eğitim verisi azaldıkça model performansı düşmüştür.

---

## 📊 Görselleştirmeler

### 🧠 EEG Kanal Bazlı Analizler
<img width="757" height="875" alt="Ekran görüntüsü 2026-05-22 174812" src="https://github.com/user-attachments/assets/4eaff350-d16d-417f-964e-7407a44869f0" />

---

### 📈 Frekans Bant Gücü Karşılaştırmaları
<img width="1536" height="850" alt="heatmap" src="https://github.com/user-attachments/assets/ce4b9c72-fb2b-45e1-b6fb-8313ec2d109e" />

---

### 📊 Değerlendirme Metrikleri
<img width="691" height="361" alt="Ekran görüntüsü 2026-05-22 163731" src="https://github.com/user-attachments/assets/b1163eb9-2192-40fa-8a02-dbf3d3f0327f" />

---

### 📉 Confusion Matrix
<img width="748" height="706" alt="Ekran görüntüsü 2026-05-22 163609" src="https://github.com/user-attachments/assets/079204e9-b12e-4e0d-9eeb-3287454c5809" />

---

## 📱 Mobil Uygulama Modülü

Bu proje yalnızca bir yapay zeka modeli değil, aynı zamanda bir uçtan uca klinik kara destek sistemidir.

Mobil uygulama:

EEG analiz sonuçlarını görselleştirir
Alzheimer / Parkinson / Healthy tahminini gösterir
Hasta bazlı geçmiş analizleri saklar
Klinik verileri kullanıcıya anlaşılır şekilde sunar
Doktorların hızlı değerlendirme yapmasını sağlar

---

🧩 Giriş Ekranı
<img width="974" height="2048" alt="WhatsApp Image 2026-05-22 at 23 06 40 (4)" src="https://github.com/user-attachments/assets/312bb7a4-7144-4669-8187-2d325f6cff9d" />

---

📊 Analiz için Veri Giriş Ekranı
<img width="973" height="2048" alt="WhatsApp Image 2026-05-22 at 23 06 40 (2)" src="https://github.com/user-attachments/assets/29c2a79f-e6ac-4f12-b4af-9fa2049abda8" />
<img width="977" height="2048" alt="WhatsApp Image 2026-05-22 at 23 06 40 (3)" src="https://github.com/user-attachments/assets/da6d85e1-51ee-4b71-8fd3-f1b2b35ad695" />

---

🧠 Model tahmin sonucu ekranı (Grafik ve istatistik görüntüleme)
<img width="973" height="2048" alt="WhatsApp Image 2026-05-22 at 23 06 40" src="https://github.com/user-attachments/assets/2753e99c-1a91-4fb5-bac2-36a6bff6041b" />
<img width="976" height="2048" alt="WhatsApp Image 2026-05-22 at 23 06 40 (1)" src="https://github.com/user-attachments/assets/f390b744-14c8-4569-9c7c-13ae504350ee" />

---

## 🛠️ Kullanılan Teknolojiler (Mobil)
- Flutter — Cross-platform mobil framework (Google)
- Dart — Flutter’ın programlama dili
- Material Design 3 — Modern UI tasarım sistemi
- fl_chart — Grafik ve bar chart görselleştirme (sonuç ekranları)
- image_picker — Galeri ve kamera erişimi
- http — REST API istekleri için HTTP client
- google_fonts — Space Grotesk ve Outfit font desteği

---

## 🚀 Gelecek Çalışmalar

- Transformer tabanlı EEG modelleri  
- Daha büyük ve dengeli veri setleri  
- Hiperparametre optimizasyonu  
- Gerçek zamanlı EEG sınıflandırma sistemi
- Mobil uygulamanın klinik seviyeye optimize edilmesi
  
---
