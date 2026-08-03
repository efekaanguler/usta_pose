# USTA — Zincirli Ortak Tahta

İki oyunculu, aynı cihazda oynanan üç paralel zincir, gizli puan ve takas oyunu.

## Yerelde açma

Dosyaları doğrudan açmak yerine klasörde küçük bir HTTP sunucusu çalıştırın:

```bash
python3 -m http.server 8080
```

Ardından `http://localhost:8080` adresini açın.

## GitHub Pages

Bu klasör bağımlılıksız statik bir sitedir. GitHub Pages için klasörü yayın kökü
olarak seçebilir veya içeriğini `docs/` klasörüne taşıyabilirsiniz. Herhangi bir
derleme adımı gerekmez.

Bu deponun GitHub Pages adresi etkinleştirildiğinde oyun şu adreste açılır:

`https://efekaanguler.github.io/usta_pose/ortak-tahta-oyunu/`

## Oyun kontrolleri

- Oyuncu panelindeki **Seç** düğmesiyle aktif oyuncuyu değiştirin.
- Her satır fiziksel olarak soldan sağa ilerler. İkinci satır soldan sağa
  6→5→4→3→2→1 dizilidir.
- İlk tasarımdaki A/B/N dağılımı bütün tahtada korunur.
- Yalnızca bu dört hücredeki Özel P1/P2 pulları sabittir; sahipleri senaryoya
  göre oyun kurulurken belirlenir ve oyun sırasında değişmez.
- Özel pullu hücrelerde şekil kısıtlaması yoktur; doğru numaralı her disk
  gelebilir. Şekiller yalnızca gizli puan hesabını etkiler.
- Normal yerleştirmelerde P1/P2 pulu bırakılmaz. Diski kimin koyduğuna
  bakılmaksızın, şekil hangi gizli kitapçıkla eşleşiyorsa puanı o oyuncu alır.
- Bir oyuncu üst üste en fazla iki disk koyabilir. İkinci yerleştirmeden sonra
  aktif oyuncu otomatik değişir; takas bu sınırı sıfırlamaz.
- **Takas yap** ile iki elden birer disk seçin; en fazla 6 takas yapılabilir.
- Gizli kitapçık kartına basarak hedefi açıp kapatın.
- Oyun tarayıcıda otomatik kaydedilir.

## Fiziksel oturumu fotoğraftan puanlama

Üst menüdeki **Fotoğraftan puanla** düğmesi fiziksel masa oturumları için ayrı
bir puan formu açar:

- Telefon kamerasından fotoğraf çekilebilir veya cihazdan JPG/PNG/WEBP seçilebilir.
- Fotoğraf hiçbir sunucuya gönderilmez; yalnızca açık tarayıcı sekmesinde tutulur.
- A/B hücrelerine fotoğrafta görülen şekil girildiğinde Senaryo 1/2 puanları
  anında hesaplanır. N hücrelerini girmek gerekmez.
- Tahta tamamlanma durumu ortak +20 bonusunu açıp kapatır.
- Sonuç panoya kopyalanabilir.
