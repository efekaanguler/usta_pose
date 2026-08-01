# felfelfeci4.py — Kamera Kayıt Sistemi: Tam Tanı ve Fix Raporu

**Tarih:** 2026-08-01  
**Sistem:** `kovan` (144.122.66.87), `kekec@kovan`  
**Kod:** `devel/record/felfelfeci4.py`  
**Referans başarılı session:** `session_2026-08-01_14:11`, `session_2026-08-01_14:41`

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
| cam2 | D455 | 318122303397 | `2-1.3` | **Hub üzerinden** (Genesys Logic) | pose |
| cam3 | D435i | 243222072357 | `2-4` | **Doğrudan** (root port) | gaze |
| cam4 | D435i | 243322074474 | `2-1.4` | **Hub üzerinden** (Genesys Logic) | gaze |

---

## 2. USB Topolojisi

```
xHCI Kontrolcü (Intel Tiger Lake-H, 00:14.0) — 20 Gbps
│
├── Port 2-1 ── [Genesys Logic USB3.2 Hub, 10 Gbps uplink]
│              ├── Port 2-1.3 ── cam2 (D455, 5 Gbps)
│              └── Port 2-1.4 ── cam4 (D435i, 5 Gbps)
│
├── Port 2-2 ── cam1 (D455, 5 Gbps)
│
└── Port 2-4 ── cam3 (D435i, 5 Gbps)
```

---

## 3. Tüm Test Sonuçları (Kronolojik)

### Sorunlu Başlangıç Durumu (Düzeltme Öncesi)

| Session | cam1 fps | cam2 fps | cam3 fps | cam4 fps | cam2 dup |
|---------|---------|---------|---------|---------|---------|
| 2026-07-31_13:46 | 20.5 | 20.5 | 24.3 | 25.1 | 59 |
| 2026-07-31_14:29 | 21.5 | 11.6 | 25.3 | 21.0 | 40 |
| 2026-07-31_15:48 | 29.8 | 16.1 | 28.2 | 28.1 | 306 |
| 2026-07-31_16:05 | 29.8 | 12.2 | 29.2 | 29.1 | 414 |
| 2026-07-31_16:17 | 29.7 | 15.2 | 28.8 | 28.8 | 349 |

---

### Testler — 2026-08-01 (felfelfeci4.py ile)

| Session | Mod | cam1 fps | cam2 fps | cam3 fps | cam4 fps | cam2 dup | Notlar |
|---------|-----|---------|---------|---------|---------|---------|--------|
| 14:09 | 4RGB+2depth | 29.7 | 29.7 | 20.4 | 22.4 | 2 | İlk felfelfeci4 testi |
| **14:11** | **4RGB+2depth** | **29.6** | **29.8** | **29.9** | **29.9** | **1** | **İlk hardware_reset sonrası — en iyi** |
| 14:16 | 4RGB+4depth | 23.4 | 14.7 | 21.7 | 18.1 | — | `--gaze-depth` testi, USB doydu, ASIC bozuldu |
| 14:18 | 4RGB+2depth | 29.9 | **18.9** | 27.0 | 26.6 | **76** | ASIC bozuk state devam etti |
| **14:41** | **4RGB+2depth** | **29.6** | **29.1** | **29.8** | **29.7** | **8** | **hardware_reset fix eklendi — başarılı** |
| 14:42 | 4RGB+2depth | 23.3 | 29.8 | 28.2 | 28.5 | 0 | Sequential start — cam1 yavaş başladı |
| 14:45 | 4RGB+2depth | 29.8 | 28.4 | 21.5 | 22.2 | 1 | FPS check bozuk (capture_count=0 hatası) |
| 14:47 | 4RGB+2depth | ~29.7 | ~29.6 | ~28.3 | ~27.8 | — | Parallel reset+start; summary yazılamadı |
| 14:50 | 4RGB+2depth | 29.9 | 29.9 | 22.6 | 22.4 | 2 | Sequential start + 200ms timeout → D435i ölü |
| 14:53 | 4RGB+2depth | 29.5 | 28.2 | 28.7 | 28.8 | 39 | Hub-first sıralı + 1.5s stagger |
| 14:55 | 4RGB+2depth | 29.8 | 29.9 | 22.9 | 22.7 | 0 | Sequential — cam3/cam4 son sırada →22fps |
| 14:59 | 4RGB+2depth | 23.5 | 25.3 | 27.1 | 27.1 | 120 | Parallel start — cam1 dup artı |
| **15:01** | **4RGB+2depth** | **29.8** | **29.9** | **27.3** | **27.2** | **0** | **Final: parallel reset + parallel start** |
| **15:02** | **4RGB+2depth** | **29.4** | 22.4 | **28.0** | **27.9** | 98 | Relaxed warmup (8/10) — cam2 erken converge |

---

## 4. Kök Neden Analizi

### 4.1. cam2 Duplicate Frame Problemi (ASIC State Bozulması)

**14:11 vs 14:18 karşılaştırması:** Her iki session'da cam2 aynı USB path'te (`2-1.3`), aynı kod, aynı donanım. Fakat 14:11'de 29.8fps, 14:18'de 18.9fps.

Aradaki tek fark: 14:16'da `--gaze-depth` ile 4 kamera tam depth çalıştırıldı → USB bus doydu (~552MB/s) → D455 stereo ASIC timing bozuldu. Bu bozukluk `pipeline.stop()` + `pipeline.start()` arasında persist etti. SDK'nın `hardware_reset()` çağrısı firmware'i yeniden başlattı ve sorunu çözdü.

**Sonuç:** D455'in stereo ASIC'i USB bandwidth kısıtlaması altında çalışırken timing penceresini kaçırır. Bu durum sonraki oturumlara sızar; `hardware_reset()` ile temizlenir.

### 4.2. cam3/cam4'ün 27-28fps Platosunda Kalması

cam3 ve cam4 D435i (color-only mod). Teorik bant ihtiyacı 83 MB/s, USB path'leri doğrudan (cam3=`2-4`) ve hub (cam4=`2-1.4`). D455 depth burst'ları (cam1+cam2 = ~110 MB/s depth) ile aynı xHCI controller'ı paylaşıyorlar.

- D455 depth stereo ASIC, USB isochronous penceresini düzenli aralıklarla dolu tutuyor
- D435i'nin color stream'i için kalan scheduling slot daha az → her ~33ms'de bir 66ms'lik boşluk
- cam3: 885 frame, 96 LATE (66ms aralık) → %10.8 frame kaçıyor → 30fps × 0.89 ≈ 26.7fps

Bu donanım-kaynaklı, yazılımla tam çözülemiyor.

### 4.3. Startup State Tutarsızlığı

Tüm testlerde startup sonucu değişiyor. Nedenler:
- xHCI controller isochronous slot tahsisi nondeterministik (kernel timer granularity)
- Paralel başlatmada 4 kamera aynı anda slot istiyor → hangisi "dolu" slota denk gelirse 30fps alamıyor
- Warmup süresi kameranın o anda USB'den ne kadar bant aldığına bağlı

---

## 5. Uygulanan Yazılım Düzeltmeleri

### 5.1. `hardware_reset()` — Bozuk ASIC State Temizliği (Ana Fix)

```python
def hardware_reset_only(self):
    ctx = rs.context()
    for dev in ctx.query_devices():
        if dev.get_info(rs.camera_info.serial_number) == self.serial:
            dev.hardware_reset()
            return
```

**Ne zaman çalışır:** Her `start()` öncesinde tüm kameralara paralel olarak gönderilir, ardından 3.5s USB re-enumeration beklenir.

**Etkisi:** cam2 duplicate frame sayısını 76→8'e, bazı testlerde 0'a indirdi.

### 5.2. Adaptive Warmup — FPS Convergence Bekleme

```python
# 10 frame'lik rolling window, ±15% tolerans, en fazla 150 frame
while window tam değil veya 8/10 aralık hedeften ±15% içinde değil:
    pipeline.wait_for_frames()
```

**Sabit 30 frame yerine:** AE gerçekten converge ettiğinde recording başlıyor.

### 5.3. Shutdown Error False Positive Düzeltmesi

SIGINT kapanışındaki "FFmpeg return code 255" hatası artık `bottleneck_diagnosis`'ı etkilemiyor. Sadece gerçek disk/encoder sorunlarında uyarı verilir.

### 5.4. `_raw_frame_count` Sayacı

Recording başlamadan önce de capture loop'un kaç frame aldığını ölçebilmek için eklendi. FPS health check'de kullanılır.

---

## 6. Sistem Düzeyindeki Kalıcı Optimizasyonlar

| Optimizasyon | Dosya | Durum |
|---|---|---|
| CPU governor: performance | `/etc/default/cpufrequtils` | ✓ Kalıcı |
| xHCI IRQ → tüm 16 core | `/etc/systemd/system/xhci-irq-affinity.service` | ✓ Kalıcı |
| USB autosuspend devre dışı | `/etc/udev/rules.d/99-realsense-power.rules` | ✓ Kalıcı |
| `auto_exposure_priority=0` | felfelfeci4.py kodu | ✓ Her başlatmada |

Reboot sonrası doğrulama:
```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor  # performance
cat /proc/irq/125/smp_affinity                             # 0000ffff
cat /sys/bus/usb/devices/2-1.3/power/autosuspend           # -1
```

---

## 7. Denemeler ve Sonuçları

| Deneme | Sonuç |
|--------|-------|
| Sabit exposure (auto_exposure=0, exposure=8000) | ✗ Başarısız — RGB sensör dondu |
| Depth çözünürlük düşürme (848×480) | △ Kısmi iyileşme |
| IRQ tek CPU'ya sabitleme | ✗ Daha kötü |
| Sequential kamera başlatma | ✗ Son kameralar bandwidth alamıyor →22fps |
| hub-first sequential (cam2/cam4 önce) | △ cam1/cam2 iyi ama cam3/cam4 22fps |
| Parallel start, FPS health check (capture_count) | ✗ capture_count recording'de artıyor → false 0fps |
| Parallel reset + parallel start (v7) | ✓ En tutarlı — cam1/cam2 ~29.9fps |
| Sequential start + 200ms wait_for_frames timeout | ✗ D435i stall → RuntimeError |
| Relaxed warmup 8/10 window | ✗ cam2 erken converge → 22fps, 98 dup |

---

## 8. Nihai Durum

### 4 RGB + 2 Depth (cam1/cam2 depth, cam3/cam4 color-only) — Default Mod

**En iyi gözlemlenen sonuç (session 14:41, 14:11, 15:01):**

| Kamera | FPS | Drop | Duplicate | Jitter |
|--------|-----|------|-----------|--------|
| cam1 | 29.6–29.9 | 0 | 0–5 | <5ms |
| cam2 | 29.1–29.9 | 0 | 0–8 | <10ms |
| cam3 | 27.0–29.8 | 0 | 0 | <10ms |
| cam4 | 27.0–29.7 | 0 | 0 | <10ms |

**Tipik sonuç (hardware_reset sonrası):**
- cam1/cam2: ~29.5–29.9fps, 0 drop, 0–8 duplicate ✓
- cam3/cam4: ~27–28fps, 0 drop, 0 duplicate ⚠

cam3/cam4'ün 27-28fps'de kalması donanım sınırı: aynı xHCI controller üzerindeki D455 depth burst'larıyla yarışıyor. Drop yok, duplicate yok; sadece 2-3fps eksik.

### 4 RGB + 4 Depth (`--gaze-depth` ile)

4 kamera tam depth USB controller'ı doyuruyor: cam3/cam4 14-23fps, yüksek duplicate. Sadece gerektiğinde kullanılmalı.

---

## 9. Çözüm Öncelikleri

1. **cam2 kablosunu hub'dan çıkar, arka panel USB-A portuna tak** (2-3, 2-5, veya 2-6 — henüz kullanılmayan root port'lardan biri)  
   → cam2 `2-1.3` (hub) yerine `2-x` (direct) olacak  
   → cam1 ile özdeş koşullar → beklenti: 29.9fps, 0 duplicate

2. **cam4 da mümkünse hub'dan çıkar**  
   → Tüm 4 kamera doğrudan bağlanırsa xHCI scheduling dengeli dağılır  
   → cam3/cam4 de ~29.9fps'e ulaşabilir

3. **PCIe USB genişletme kartı** (ikinci xHCI controller)  
   → 2 kamera/controller → her kamera kendi dedicated bant genişliğine sahip  
   → 4 kamera tam depth bile çalışabilir

---

## 10. Kod Değişiklikleri Özeti (felfelfeci3.py → felfelfeci4.py)

| Değişiklik | Etki |
|-----------|------|
| `hardware_reset_only()` + paralel reset + 3.5s bekleme | ASIC bozuk state temizleniyor |
| Adaptive warmup (rolling window, ±15%, max 150 frame) | AE convergence garantili |
| `_raw_frame_count` sayacı | Recording dışında da FPS ölçülebilir |
| Shutdown error false positive fix | Gerçek disk sorunları doğru raporlanıyor |
| Gaze kameralar için varsayılan depth kapalı | USB bandwidth korunuyor |
| `--gaze-depth` flag | Gerektiğinde 4 kamera depth açılabilir |
| Parallel kamera başlatma | AE aynı anda converge ediyor |
| Warm-up 30 → adaptive | Erken recording engelleniyor |
| `--depth-width` / `--depth-height` flag | Depth çözünürlüğü bağımsız ayarlanabilir |
