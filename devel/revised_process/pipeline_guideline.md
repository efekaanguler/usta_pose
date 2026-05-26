# Yeni Nesil USTA Poz İşleme ve Veri Seti Hattı (Revised Pipeline)

Bu doküman, çoklu kamera sistemlerinden alınan video kayıtlarını makine öğrenmesi modelleri (zaman serisi analizi, sınıflandırma vb.) için en uygun hale getiren yeni veri işleme mimarisini açıklamaktadır.

## 1. Neden Yeni Bir Pipeline'a İhtiyaç Duyduk?

Eski sistem (`process/` klasörü altındaki yaklaşım), tüm kameraları ortak bir zaman damgasına (master frame) "en yakın (nearest neighbor)" eşleştirme yöntemiyle zorluyordu.
- **Problem:** Kameralardan biri 24 FPS, diğeri 30 FPS hızında kayıt aldığında, en yakın kareyi seçme mantığı bazı kareleri **çoğaltıyor** veya **atlıyordu**. Bu da iskeletlerde mikro-donmalara ve sıçramalara (temporal jitter) yol açıyordu. Zaman serisi modelleri (RNN, LSTM, Transformer) bu durumu bir hareket (aksiyon) olarak algılayıp yanılabilir.
- **Problem 2:** Sensör hatalarından kaynaklı (Depth kamerasının okuyamaması) `NaN` değerleri doğrudan modele gürültü olarak yansıyordu.
- **Çözüm:** Eşleştirme yapmak yerine **Sürekli Yeniden Örnekleme (Continuous Resampling)** yaklaşımına geçilmiştir.

Ek olarak, makine öğrenmesi algoritmaları doğrudan global `(X,Y,Z)` noktaları ile eğitilmeye yatkın değildir. Kişinin odanın/masanın neresinde durduğu, kişinin yaptığı eylemi (jestler, pozlar) etkilememelidir. Bu yüzden **Root-Relative (Kök-Göreceli)** bir temsile (representation) ihtiyaç vardır.

---

## 2. Mimari ve İş Akışı

Yeni boru hattı (pipeline) 3 temel adımdan oluşmaktadır:

### Adım 1: Bağımsız Veri Çıkarımı (Independent Extraction)
Kameraların kareleri birbiriyle eşleştirilmez. Her kamera kendi donanım zaman damgasına (`hw_timestamp_ms`) göre bağımsız olarak işlenir.
- **İşlem:** Her kare için varsayılan olarak RTMPose-L 2D (Cam1, Cam2) ve Gaze tahmini (Cam3, Cam4) çalıştırılır. Pose modelinin görüntü düzlemindeki noktaları RealSense depth ile metrik 3D'ye çevrilir ve sadece kameranın kendi lokal uzayında (Cam Coordinates) bırakılır. Global uzaya dönüşüm burada YAPILMAZ. RTMW3D yalnızca `POSE_MODEL=rtmw3d` ile açıkça seçilmelidir; metrik depth dönüşümü için geniş/tam-kare RTMW3D kutuları engellenir, sıkı kişi bbox'u verilmelidir. Cam1 varsayılan olarak görüntünün sağdaki 1/5 bölümünü kırpar ve yakın/çok büyük foreground iskeletlerini reddeder.
- **Çıktı:** `cam1_pose_raw.csv`, `cam2_pose_raw.csv`... (Her kameranın kendi FPS'ine sahip lokal ham verileri)

### Adım 2: Sürekli Yeniden Örnekleme (Continuous Resampling - Adaptive FPS)
Farklı frekanslardaki kameraları aynı eksene oturtmak için kontrollü interpolasyon uygulanır.
- **Zaman Hizalaması:** Çıkış FPS'i kullanılabilir kamera zaman damgalarından tahmin edilir; artık kör şekilde 30 FPS'e zorlanmaz.
- **İnterpolasyon (Imputation):** Sadece kısa boşluklar (varsayılan 150 ms) lineer interpolasyon ile doldurulur. İlk geçerli gözlemden önce, son geçerli gözlemden sonra veya uzun kayıp bölgelerde değer uydurulmaz.
- **Filtreleme:** **Savitzky-Golay filtresi** sadece kesintisiz ve finite segmentlerin içinde uygulanır; NaN boşluklarının üzerinden smoothing yapılmaz.

### Adım 3: Koordinat Dönüşümü ve Root-Relative Dışa Aktarma
Zaman senkronu sağlanmış ve filtrelenmiş noktalar, makine öğrenmesine (ML) uygun formata dönüştürülür.
- **Ortak Frame (World) Matematiği:** OpenCV kalibrasyonundan gelen dış (extrinsics) parametreler NPZ içinde `ref -> cam` olarak saklanır; işleme sırasında ters çevrilerek `P_ref = R_cam_to_ref @ P_cam + t_cam_to_ref` uygulanır. Böylece tüm kameralardaki noktalar hatasız bir şekilde Cam1'in referans eksenine (World) taşınır.
- **Root-Relative İşlemi:** Root artık yalnızca kalça ortasına bağlı değildir. Öncelik plausible kalçalardadır; kalçalar eksik veya güvenilmezse torso ortalaması, o da yoksa omuz ortası kullanılır. `root_source` kodları `0=missing`, `1=hips`, `2=torso`, `3=shoulders` olarak saklanır. Geriye kalan tüm noktalar bu root'a olan vektörel uzaklığı (offset) şeklinde kaydedilir.
- **Dışa Aktarma (.parquet):** Tüm veri tablo halinde makine öğrenmesi standartı olan bir Parquet dosyasına yazılır.

---

## 3. Kullanılan Teknolojilerin Artıları, Eksileri ve Alternatifleri

### A. Root-Relative (Kök-Göreceli) Koordinatlar
- **[+] Artıları:** İnsan beden hareketini (pose), kişinin konumundan (translation) izole eder (Translation invariance). Model sadece hareketin kendisine odaklanır.
- **[-] Eksileri:** İki kişinin birbiriyle olan mesafesini/etkileşimini ölçmek model için bir adım daha zorlaşır (çünkü iki farklı root'un farkını hesaplaması gerekir).

### B. Zaman Serisi İnterpolasyonu ve Savitzky-Golay Filtresi
- **[+] Artıları:** Mükemmel zaman senkronizasyonu sağlar. FPS düşüşleri veya donanım gecikmeleri (lag) elimine edilir. Noktalar akıcı ve pürüzsüz (smooth) bir şekilde akar. Modelin güvenilmez dediği yerler kurtarılır.
- **[-] Eksileri:** Eski sınırsız interpolasyon hayali hareket üretebiliyordu. Yeni sürüm uzun boşlukları NaN bırakır; model eğitimi sırasında bu maskelerin dikkate alınması gerekir.

### C. Depolama: .parquet Formatı
- **[+] Artıları:** CSV'ye göre sıkıştırma oranı 10 kata kadar daha iyidir. Binary olduğu için Pandas ile okuma/yazma işlemleri milisaniyeler sürer. Veri tiplerini otomatik tanır.
- **[-] Eksileri:** Metin editöründe doğrudan açılarak gözle (insan tarafından) okunamaz. Okumak için Python (Pandas) vb. araçlar gerekir.
