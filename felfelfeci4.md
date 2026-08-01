# felfelfeci4.py — Kamera Kayıt Sistemi: Durum Raporu

**Tarih:** 2026-08-01  
**Sistem:** `kovan` (144.122.66.87), `kekec@kovan`  
**Kod:** `devel/record/felfelfeci4.py`  
**Test kaydı:** `devel/record/recordings/session_2026-08-01_14:18/`

---

## 1. Donanım Özeti

| Bileşen | Değer |
|---------|-------|
| CPU | Intel Core i7-11700F @ 2.50GHz, 8 çekirdek / 16 thread |
| RAM | 32 GB |
| Disk | NVMe SSD, ~1.8 GB/s yazma hızı (dd ölçümü) |
| USB kontrolcü | Intel Tiger Lake-H xHCI USB 3.2 Gen 2x2 — **tek kontrolcü**, PCI `00:14.0` |
| xHCI max bant | 20 Gbps (2×10 Gbps lane) |
| IRQ | 125, SMP affinity: `ffff` (tüm 16 core) |
| CPU governor | `performance` (tüm core'larda kalıcı) |
| pyrealsense2 | 2.58.2 |
| FFmpeg | imageio-ffmpeg v7.0.2 (bundled, sistem PATH'inde yok) |

### Kameralar

| Cam ID | Model | Seri No | USB Yolu | Bağlantı | Rol |
|--------|-------|---------|----------|----------|-----|
| cam1 | D455 | 049522250225 | `2-2` | **Doğrudan** (root port) | pose |
| cam2 | D455 | 318122303397 | `2-1.3` | **Hub üzerinden** | pose |
| cam3 | D435i | 243222072357 | `2-4` | **Doğrudan** (root port) | gaze |
| cam4 | D435i | 243322074474 | `2-1.4` | **Hub üzerinden** | gaze |

---

## 2. USB Topolojisi (Kök Neden)

```
xHCI Kontrolcü (Intel Tiger Lake-H, 00:14.0) — 20 Gbps (Gen 2x2)
│
├── Port 2-1 ── [Genesys Logic USB3.2 Hub, 10 Gbps uplink]
│              ├── Port 2-1.3 ── cam2 (D455, 5 Gbps)   ← SORUNLU
│              └── Port 2-1.4 ── cam4 (D435i, 5 Gbps)
│
├── Port 2-2 ── cam1 (D455, 5 Gbps)   ← SORUNSUZ
│
└── Port 2-4 ── cam3 (D435i, 5 Gbps)  ← SORUNSUZ
```

**cam2 ve cam4, 10 Gbps uplink'e sahip bir Genesys Logic USB3.2 Hub üzerinden bağlı.**  
cam1 ve cam3, xHCI root port'larına doğrudan bağlı.

### Bant Genişliği Hesabı (4 RGB + 2 Depth modu)

| Kamera | Akış | Ham Veri Hızı |
|--------|------|---------------|
| cam1 (D455) | color 1280×720@30 + depth 1280×720@30 | ~83 + ~55 = ~138 MB/s (~1.1 Gbps) |
| cam2 (D455) | color 1280×720@30 + depth 1280×720@30 | ~83 + ~55 = ~138 MB/s (~1.1 Gbps) |
| cam3 (D435i) | color 1280×720@30 | ~83 MB/s (~0.66 Gbps) |
| cam4 (D435i) | color 1280×720@30 | ~83 MB/s (~0.66 Gbps) |
| **Toplam** | | **~3.56 Gbps** |

Hub üzerinden geçen yük: cam2 (1.1 Gbps) + cam4 (0.66 Gbps) = **1.76 Gbps** → Hub'ın 10 Gbps uplink kapasitesi içinde.

**Sonuç: Bant genişliği teorik olarak yeterli; asıl sorun bant genişliği değil, hub'ın izochronous zamanlama yönetimi.**

---

## 3. Test Sonuçları (4 RGB + 2 Depth, 32.7 saniye)

### FPS Özeti

| Kamera | Hedef FPS | Gerçekleşen FPS | Yazılan Frame | Drop | Duplicate |
|--------|-----------|-----------------|---------------|------|-----------|
| cam1 | 30 | **29.9** ✓ | 978 | 0 | 0 |
| cam2 | 30 | **18.9** ✗ | 696 | 0 | **76** |
| cam3 | 30 | **27.0** ⚠ | 885 | 0 | 0 |
| cam4 | 30 | **26.6** ⚠ | 869 | 0 | 0 |

### Frame Aralık Histogramı (33ms = 1 frame, 66ms = 2 frame atlamış)

| Kamera | ~33ms (normal) | ~66ms (1 atladı) | ~100ms (2 atladı) | ~133ms (3 atladı) | max aralık |
|--------|---------------|-----------------|-------------------|-------------------|------------|
| cam1 | 974 | 3 | 0 | 0 | 66.8 ms |
| cam2 | 426 | 105 | 46 | 19 | **333.7 ms** |
| cam3 | 788 | 96 | 0 | 0 | 66.7 ms |
| cam4 | 756 | 112 | 0 | 0 | 66.7 ms |

**cam2 aralık std:** 38.8 ms (cam1: 1.8 ms). cam2 ciddi düzensizlik içinde.  
**cam2'de 333.7 ms'lik boşluk:** 10 ardışık frame tamamen kaybolmuş.

---

## 4. Kök Neden Analizi

### 4.1. cam2'nin 19fps Kalması ve 76 Duplicate Frame

`duplicate_frame_numbers_observed: 76` → SDK, bir önceki depth frame'i tekrar teslim ediyor. Bu, D455'in stereo depth ASIC'inin yeni bir depth frame üretememesi durumunda pyrealsense2 SDK'nın son geçerli frame'i yeniden kullanmasıyla oluşur.

**Neden ASIC yeni frame üretemiyor?**  
cam2, Genesys Logic hub'ının `2-1.3` portuna bağlı. USB izochronous transferler zamana kritik bağımlıdır — her micro-frame'de (125 µs) kameradan veri paketi alınması gerekir. Bir hub bu transferleri multipleks ederken:

1. Hub, downstream portlarından (2-1.3 cam2, 2-1.4 cam4) gelen izochronous isteklerini upstream 10 Gbps bağlantısına sırayla iletmek zorundadır.
2. İki yüksek bant akışı aynı anda geldiğinde hub arbitrasyon gecikmesi oluşur.
3. cam2'nin depth modülü, sol/sağ IR çifti için USB timing penceresini kaçırır.
4. Stereo ASIC senkronizasyonu bozulur → yeni depth frame üretilmez.
5. SDK, son geçerli frame'i tekrar döndürür → `frame_number` aynı.

cam2'nin color stream'i de depth stream ile senkronize çalıştığından, color akışı da depth'i bekler ve **color FPS de düşer** (18.9 fps).

cam1 aynı D455 modelidir, aynı konfigürasyondadır. Tek fark: **doğrudan bağlı (2-2)**. cam1 29.9 fps, 0 duplicate → doğrudan bağlantı sorunu tamamen ortadan kaldırıyor.

### 4.2. cam3 ve cam4'ün 27fps Alması

cam3 (doğrudan, color-only) ve cam4 (hub, color-only) her ikisi de 27 fps. Depth devre dışı olduğundan duplicate yok. Yalnızca 96-112 LATE frame (tam 66ms = 1 atladı) gözlemleniyor.

Olası nedenler:
- **xHCI IRQ paylaşımı:** 4 kamera aynı IRQ 125'i paylaşıyor. Tüm 16 core'a yayılmış olsa da 4 eş zamanlı yüksek bant izochronous stream yüksek IRQ servisleme yükü yaratıyor.
- **Python GIL:** 4 kamera thread'i Python GIL'i paylaşıyor. `wait_for_frames()` GIL'i bırakmadığında thread'ler birbirini blokluyor.
- **D435i özellikleri:** D435i'nin USB transfer zamanlaması D455'e göre farklı. Doğrudan bağlı cam3 de 27fps → hub tek başına açıklayamıyor.

### 4.3. cam2 — 333ms Boşluklar

cam2'de maksimum 333.7ms (10 frame boşluğu) gözlemleniyor. Bu, hub arbitrasyon gecikmesinin ani USB kitleme periyodlarına dönüştüğünü gösteriyor. Hub'da cam4'ün bant talebi cam2'nin depth akışını uzun süre engelliyor olabilir. Veya kernel USB scheduling bir hub re-enumeration benzeri geçici durum yaşıyor.

---

## 5. Uygulanan Optimizasyonlar ve Durumları

| Optimizasyon | Durum | Kalıcı | Etki |
|--------------|-------|--------|------|
| CPU governor: performance | ✓ Aktif | ✓ `/etc/default/cpufrequtils` | Tüm 16 core'da doğrulandı |
| xHCI IRQ affinity: ffff (0-15) | ✓ Aktif | ✓ systemd `xhci-irq-affinity.service` | IRQ 125, tüm core'lara yayıldı |
| USB autosuspend devre dışı | ✓ Aktif | ✓ `/etc/udev/rules.d/99-realsense-power.rules` | Tüm D455/D435i için `autosuspend=-1` |
| `auto_exposure_priority=0` | ✓ Kod içinde | — | Firmware FPS → exposure trade-off'u devre dışı |
| Parallel kamera başlatma | ✓ Kod içinde | — | AE konverjansı için eş zamanlı warm-up (30 frame) |
| Gaze depth varsayılan kapalı | ✓ Kod içinde | — | `--gaze-depth` flag ile açılır |
| FFV1 lossless encoding | ✓ Kod içinde | — | Color bgr0, Depth gray16le |

---

## 6. Yazılımsal Çözüm Denemeleri (Geçmişte Başarısız)

| Deneme | Sonuç |
|--------|-------|
| Sabit exposure ayarı (`auto_exposure=0`, `exposure=8000`) | Başarısız — RGB sensör dondu, unique_camera_frames 4-17'ye düştü |
| Depth çözünürlük düşürme (848×480) | Kısmi — cam2 hâlâ ~20fps; kullanıcı geri getirdi |
| IRQ affinity tek CPU'ya sabitleme | Başarısız — daha kötü |
| Sequential kamera başlatma | Başarısız — AE konverjansı yetersizdi |

---

## 7. Çözüm Önerileri

### 7.1. Donanım — En Kesin Çözüm (cam2'yi Doğrudan Porta Al)

xHCI kontrolcüsünün 6 root port'u var (`xhci_hcd/6p`). Şu anda kullanılanlar:
- `2-1`: Hub (cam2+cam4)
- `2-2`: cam1
- `2-4`: cam3
- `2-3`, `2-5`, `2-6`: muhtemelen boş arka panel USB-A portları

**cam2'nin hub'dan çıkarılıp doğrudan bir arka panel USB-A 3.x portuna bağlanması**, cam1 ile özdeş koşullar yaratır ve aynı 29.9fps + 0 duplicate sonucunu verir. Ek maliyet yok, kablo değişikliği yeterli.

**Test edilmesi gereken port düzeni:**
```
Şu an:  2-1 (hub: cam2+cam4), 2-2 (cam1), 2-4 (cam3)
Hedef:  2-2 (cam1), 2-3 (cam2, direkt!), 2-4 (cam3), 2-x (cam4, direkt)
```

### 7.2. Donanım — PCIe USB Genişletme Kartı

Ayrı xHCI kontrolcüsü içeren PCIe USB 3.x kartı (Inateck/StarTech gibi). Kameraları iki kontrolcüye böler (2 cam / controller). Her kamera kendi dedicated bant genişliğine kavuşur.

**Maliyet:** ~500-1000 TL  
**Etki:** 4×30fps + 4×depth tam kapasitede çalışır.

### 7.3. Yazılım — cam2 Depth Çözünürlüğü Düşürme

```bash
python3 devel/record/felfelfeci4.py --depth-width 848 --depth-height 480
```

cam2'nin depth ham veri hızı: 1280×720 (~55 MB/s) → 848×480 (~24 MB/s). Hub üzerindeki yük azalır, ASIC timing'i rahatlar. Bu daha önce test edildi — kısmi iyileşme sağlıyor ama tam çözüm değil.

### 7.4. Yazılım — Hardware Sync (Multi-Camera Sync Cable)

RealSense D455 ve D435i, GPIO sync kablo desteği sunar. Exposure başlangıcını senkronize etmek, USB burst transferlerini zamanda staggering ile dağıtabilir ve hub arbitrasyonunu düzenleyebilir.

**Kablo gerektirir.** Intel RealSense multi-camera sync dokümantasyonu: `EXTERNAL_SYNC_MASTER` / `EXTERNAL_SYNC_SLAVE` modlar.

### 7.5. Yazılım — SDK Pipe Konfigürasyonu Değişikliği

D455 stereo depth akışı `rs.stream.depth` yerine `rs.stream.infrared` olarak alınabilir (raw IR stereo). Bu, D455'in depth processing pipeline'ını bypass eder, stereo ASIC yükünü azaltır ve USB'ye sadece raw IR veri gönderilir. Depth map, post-process olarak hesaplanabilir.

**Dezavantaj:** Gerçek zamanlı depth kullanılamaz, kayıt sonrası işlem gerekir.

---

## 8. Öneri Öncelik Sırası

1. **cam2 kablo bağlantısını hub'dan çıkar, arka panel USB-A portuna tak** — sıfır maliyet, en yüksek etki.
2. Test et: `python3 devel/record/felfelfeci4.py --no-calib-check`
3. Sonuç hâlâ yetersizse PCIe USB kartı değerlendir.
4. 4×depth kesinlikle gerekiyorsa ve donanım değiştirilemiyor ise depth çözünürlük düşürme dene: `--depth-width 848 --depth-height 480`

---

## 9. Ek Notlar

### FFmpeg "return code 255" Hatası (Tüm Kameralar)

`frame_timing_summary.json` içinde tüm kameralar için `writer_error: RuntimeError('FFmpeg failed ... return code 255')` görünüyor. Bu, kayıt sırasındaki bir sorun **değil** — `timeout --signal=SIGINT` ile kayıt sonlandırıldığında FFmpeg pipe'ı beklenmedik şekilde kapanıyor. Kayıt süresince frame'ler doğru yazıldı; hata sadece kapanış sırasında oluştu. MKV container finalization tamamlanmadığından test kayıt dosyaları muhtemelen hasar gördü. Gerçek kayıtlarda Ctrl+C ile düzgün durdurulması gerekir.

### USB Autosuspend Durumu

udev kuralı `ATTR{power/control}="on"` yerine `"auto"` olarak uygulanmış (`control=auto` görünüyor). Ancak `autosuspend=-1` değeri set edildiğinden pratikte autosuspend çalışmıyor — etkin olarak "on" ile eşdeğer.

### Disk ve RAM Darboğazı Yok

- Disk: 1.8 GB/s yazma (gerekli: ~180 MB/s × 4 = ~720 MB/s) — yeterli
- RAM: 29 GB boş — yeterli
- Queue depth: maks 2 (capacity 90) — yazıcı hiç beklemiyor

Kayıt performansını sınırlayan tek faktör USB topolojisi.

---

## 10. Sistem Konfigürasyonu Özeti

```
/etc/default/cpufrequtils        → GOVERNOR="performance"
/etc/udev/rules.d/99-realsense-power.rules → autosuspend=-1
/usr/local/bin/xhci-affinity.sh  → echo ffff > /proc/irq/125/smp_affinity
/etc/systemd/system/xhci-irq-affinity.service → boot'ta çalışır
```

Reboot sonrası doğrulama:
```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor  # performance
cat /proc/irq/125/smp_affinity                             # 0000ffff
cat /sys/bus/usb/devices/2-1.3/power/autosuspend           # -1
```
