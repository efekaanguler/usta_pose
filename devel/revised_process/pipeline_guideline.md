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
- **İşlem:** Her kare için RTMPose (Cam1, Cam2) ve Gaze tahmini (Cam3, Cam4) çalıştırılır. Noktaların 3D Derinlik projeksiyonu sadece kameranın kendi lokal uzayında (Cam Coordinates) bırakılır. Global uzaya dönüşüm burada YAPILMAZ.
- **Çıktı:** `cam1_pose_raw.csv`, `cam2_pose_raw.csv`... (Her kameranın kendi FPS'ine sahip lokal ham verileri)

### Adım 2: Sürekli Yeniden Örnekleme (Continuous Resampling - 30 FPS)
Farklı frekanslardaki kameraları aynı eksene oturtmak için interpolasyon uygulanır.
- **Zaman Hizalaması:** Başlangıç zamanından bitiş zamanına kadar her 33.333 milisaniyede (30 FPS) bir sanal zaman damgası (timestamp) oluşturulur. 
- **İnterpolasyon (Imputation):** Model güvenilirliğinin (confidence) çok düşük olduğu veya Depth sensöründen okunamayan kısımlar (`NaN`) spline/lineer interpolasyon ile doldurularak, modelin hareketi kesintisiz algılaması sağlanır.
- **Filtreleme:** `scipy.interpolate` ile hizalanan noktalar **Savitzky-Golay filtresi**'nden geçirilerek sinyaldeki mikro titremeler ve gürültüler (jitter) temizlenir.

### Adım 3: Koordinat Dönüşümü ve Root-Relative Dışa Aktarma
Zaman senkronu sağlanmış ve filtrelenmiş noktalar, makine öğrenmesine (ML) uygun formata dönüştürülür.
- **Ortak Frame (World) Matematiği:** OpenCV kalibrasyonundan gelen dış (extrinsics) parametreler NPZ içinde `ref -> cam` olarak saklanır; işleme sırasında ters çevrilerek `P_ref = R_cam_to_ref @ P_cam + t_cam_to_ref` uygulanır. Böylece tüm kameralardaki noktalar hatasız bir şekilde Cam1'in referans eksenine (World) taşınır.
- **Root-Relative İşlemi:** Kişinin ana gövdesi olan **Pelvis (Kalça)** noktasının Cam1'e olan mutlak pozisyonu (`root_x, root_y, root_z`) saklanır. Geriye kalan tüm noktalar (baş, kollar, eller), bu pelvise olan vektörel uzaklığı (offset) şeklinde kaydedilir. Örn: `kpt_hand_x = kpt_global_hand_x - root_x`.
- **Dışa Aktarma (.parquet):** Tüm veri tablo halinde makine öğrenmesi standartı olan bir Parquet dosyasına yazılır.

---

## 3. Kullanılan Teknolojilerin Artıları, Eksileri ve Alternatifleri

### A. Root-Relative (Kök-Göreceli) Koordinatlar
- **[+] Artıları:** İnsan beden hareketini (pose), kişinin konumundan (translation) izole eder (Translation invariance). Model sadece hareketin kendisine odaklanır.
- **[-] Eksileri:** İki kişinin birbiriyle olan mesafesini/etkileşimini ölçmek model için bir adım daha zorlaşır (çünkü iki farklı root'un farkını hesaplaması gerekir).

### B. Zaman Serisi İnterpolasyonu ve Savitzky-Golay Filtresi
- **[+] Artıları:** Mükemmel zaman senkronizasyonu sağlar. FPS düşüşleri veya donanım gecikmeleri (lag) elimine edilir. Noktalar akıcı ve pürüzsüz (smooth) bir şekilde akar. Modelin güvenilmez dediği yerler kurtarılır.
- **[-] Eksileri:** Eğer kamera saniyelerce (çok uzun süre) donmuşsa, interpolasyon orada gerçekte olmayan "hayali" ve pürüzsüz bir kayma hareketi uydurur.

### C. Depolama: .parquet Formatı
- **[+] Artıları:** CSV'ye göre sıkıştırma oranı 10 kata kadar daha iyidir. Binary olduğu için Pandas ile okuma/yazma işlemleri milisaniyeler sürer. Veri tiplerini otomatik tanır.
- **[-] Eksileri:** Metin editöründe doğrudan açılarak gözle (insan tarafından) okunamaz. Okumak için Python (Pandas) vb. araçlar gerekir.
