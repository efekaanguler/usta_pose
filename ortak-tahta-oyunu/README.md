# USTA — Ortak Tahta

İki oyunculu, aynı cihazda oynanan interaktif gizli hedef ve takas oyunu.

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
- Aktif oyuncunun elinden bir diske, ardından aynı numaralı boş hücreye basın.
- **Takas yap** ile iki elden birer disk seçin; en fazla 6 takas yapılabilir.
- Gizli kitapçık kartına basarak hedefi açıp kapatın.
- Oyun tarayıcıda otomatik kaydedilir.
