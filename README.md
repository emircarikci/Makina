# Makina

Makine Öğrenmesi Projesi: Kalp Hastalığı Tahmini Bu proje, makine öğrenmesi algoritmaları kullanılarak kalp hastalığı olasılığını tahmin etmeyi amaçlamaktadır. Veri seti, yaş, kolesterol ve cinsiyet gibi çeşitli özellikleri içermektedir ve bu veriler, çeşitli sınıflandırma algoritmalarıyla analiz edilmiştir. Hedef, kalp hastalığını tahmin etmek için en doğru modeli belirlemektir.

Proje Adımları

Veri Yükleme ve Ön İşleme heart.csv veri seti pandas kullanılarak yüklendi ve eksik veriler kontrol edilerek gerekli işlemler yapıldı. Veri setinin temel istatistikleri görüntülenerek veri hakkında genel bir fikir edinildi.

Eksik veri tespiti Eksik veri olup olmadığı kontrol edildi Özellik Ölçekleme: Yaş ve kolesterol gibi özellikler, model performansını iyileştirmek için StandardScaler ile standartlaştırıldı.

Model Geliştirme Veri seti eğitim ve test setlerine (yüzde 80 eğitim, yüzde 20 test) ayrıldı.

Aşağıdaki makine öğrenmesi modelleri kullanıldı:

Lojistik Regresyon Destek Vektör Makinesi (SVM) Random Forest Classifier Karar Ağacı Classifier K-Nearest Neighbors (KNN)

Hiperparametre Optimize Her modelin hiperparametrelerini optimize etmek için GridSearchCV kullanıldı. Her model için en iyi parametreler seçildi.

Model Değerlendirme Model performansını değerlendirmek için aşağıdaki metrikler kullanıldı:

Başarı Oranı (Accuracy) F1 Skoru ROC AUC Skoru Precision ve Recall Ek olarak, her model için sınıflar arasındaki performansı görselleştirmek amacıyla confusion matrix'ler oluşturuldu.

Sonuçlar En Başarılı Model: Random Forest Classifier en yüksek başarı oranı ve F1 skoru ile kalp hastalığı tahmininde en iyi performansı sergilemiştir. Model Karşılaştırması: Modellerin başarı oranı, F1 skoru, ROC AUC, precision ve recall gibi metriklere göre karşılaştırılması yapılır.

Görselleştirmeler Korelasyon Matrisi: Özellikler arasındaki ilişkileri görselleştiren bir ısı haritası oluşturur. Öğrenme Eğrisi: Eğitim verisi büyüklüğü arttıkça modelin performansını gösteren bir öğrenme eğrisi çizilmiştir ve arttıkça test skorumuz artıyor. Sonuç Random Forest Classifier, kalp hastalığını tahmin etmek için en güvenilir performansı sergilemiş ve bu proje için tercih edilen model olmuştur.

Ad: Emir Çarıkçı

Numara: 21360859060

Youtube: https://youtu.be/6W9dwhywY8c

Python Sertifikası:[Sıfırdan_İleri_Seviye_Python_Programlama_Sertifika.pdf](https://github.com/user-attachments/files/18469838/Sifirdan_Ileri_Seviye_Python_Programlama_Sertifika.pdf)
