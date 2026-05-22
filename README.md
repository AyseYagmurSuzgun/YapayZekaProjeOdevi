🧠 EEG Tabanlı Alzheimer ve Parkinson Sınıflandırma (Multimodal Yapay Zeka)
📌 Proje Tanımı

Bu projede, EEG (Elektroensefalografi) sinyalleri ve klinik veriler kullanılarak Alzheimer, Parkinson ve sağlıklı bireylerin sınıflandırılması amacıyla bir yapay zeka modeli geliştirilmiştir.

Model, beyin sinyallerinden elde edilen frekans bant güçlerini (delta, teta, alfa, beta, gama) ve klinik ölçümleri (MMSE, MoCA, UPDRS, yaş, cinsiyet) birlikte kullanarak multimodal bir sınıflandırma yapısı oluşturmaktadır.

🎯 Projenin Amacı

Bu çalışmanın amacı, invaziv olmayan EEG sinyalleri ve klinik verileri kullanarak nörodejeneratif hastalıkların erken teşhisinde kullanılabilecek bir karar destek sistemi geliştirmektir.

📊 Veri Seti
Alzheimer veri seti: ds004504 (OpenNeuro / Kaggle)
Parkinson veri seti: ds004584 (OpenNeuro / Kaggle)
Toplam katılımcı sayısı: ~237 birey
EEG Kanalları

Veri setleri farklı kanal sistemleri kullandığı için ortak kanal eşleştirmesi yapılmıştır. Kullanılan 14 ortak kanal:

P4, P3, O2, O1, Fz, Fp2, Fp1, F8, F7, F4, F3, Cz, C4, C3

🧪 Sınıflar
0 → Healthy (Sağlıklı)
1 → Alzheimer
2 → Parkinson
⚙️ Veri Ön İşleme

Veri üzerinde aşağıdaki ön işleme adımları uygulanmıştır:

1–40 Hz band-pass filtreleme
Ortalama referanslama
ICA (artefakt temizleme)
2 saniyelik sabit epoch segmentasyonu
Z-score normalizasyonu
Kanal eşleştirme ve standardizasyon
🧠 Özellik Çıkarımı
EEG Tabanlı Özellikler:
Delta bant gücü
Teta bant gücü
Alfa bant gücü
Beta bant gücü
Gama bant gücü
Klinik Özellikler:
MMSE
MoCA
UPDRS
Yaş
Cinsiyet

Tüm bu özellikler birleştirilerek multimodal giriş vektörü oluşturulmuştur.

🏗️ Model Mimarisi

Projede hibrit bir derin öğrenme modeli kullanılmıştır:

EEG Modülü
1D CNN (özellik çıkarımı)
Attention mekanizması (önemli sinyal bölgelerine odaklanma)
Klinik Veri Modülü
Çok katmanlı yapay sinir ağı (MLP)
Birleştirme (Fusion)
EEG ve klinik özelliklerin birleştirilmesi
Son sınıflandırma katmanı (3 sınıf çıktı)
🧪 Eğitim Detayları
Framework: PyTorch
Epoch: 25
Optimizer: Adam
Loss fonksiyonu: Focal Loss
Sınıf dengesizliği: Class weight yöntemi
📊 Değerlendirme Metrikleri

Model performansı aşağıdaki metriklerle değerlendirilmiştir:

Accuracy (Doğruluk)
Precision (Kesinlik)
Recall (Duyarlılık)
F1-score
Confusion Matrix
📈 Sonuçlar
Accuracy: ~%63
Macro F1-score: ~0.63
Gözlemler:
Parkinson sınıfında yüksek recall değeri elde edilmiştir
Alzheimer sınıfında daha yüksek precision gözlemlenmiştir
Healthy sınıfında sınıflar arası benzerlik nedeniyle performans düşüşü görülmüştür
📉 Deneyler

Farklı veri bölme oranları test edilmiştir:

80/20
70/15/15
60/20/20
50/25/25

Sonuç: Eğitim verisi azaldıkça model performansında düşüş gözlemlenmiştir.

📊 Görselleştirmeler

Projede aşağıdaki analizler yapılmıştır:

EEG kanal bazlı sinyal analizleri
Frekans bant gücü karşılaştırmaları
Confusion matrix
Eğitim loss grafikleri
🚀 Gelecek Çalışmalar
Transformer tabanlı EEG modelleri
Daha büyük ve dengeli veri setleri
Hiperparametre optimizasyonu (Grid / Random Search)
Gerçek zamanlı EEG sınıflandırma sistemi
