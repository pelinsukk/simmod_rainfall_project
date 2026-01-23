# SimMod Rainfall Project — Demo Baseline (Radar → Ground Rainfall)
# SimMod Yağış Projesi — Demo Baseline (Radar → Yer Yağışı)

This repository contains a small demo pipeline that pairs 3D radar reflectivity snapshots with 10-minute station precipitation labels and trains a very simple baseline model.  
(Bu repo, 3B radar reflektivite anlarını 10 dakikalık istasyon yağış etiketleriyle eşleyip çok basit bir baseline model eğiten küçük bir demo pipeline içerir.)

The goal of this demo is to prove the data loading + time matching + feature extraction workflow works end-to-end with the available months (including month 12).  
(Bu demonun amacı, veri yükleme + zaman eşleme + özellik çıkarma akışının mevcut aylarla (12. ay dahil) uçtan uca çalıştığını göstermektir.)

This is not the final model, only a sanity-check baseline to unblock the next steps.  
(Bu nihai model değil, sonraki adımları açmak için bir “sanity-check” baseline’dır.)

---

## What the demo does
## Demo ne yapıyor

1) **Radar file discovery (recursive)**  
(1) **Radar dosyası bulma (rekürsif)**  
- Scans `radar/` recursively and collects files ending with `*hd5` (no dot) plus common `h5/hdf5` patterns.  
  (`radar/` altında rekürsif arama yapar ve `*hd5` (noktasız) ile biten dosyaları ve yaygın `h5/hdf5` pattern’larını toplar.)

2) **Label input**  
(2) **Etiket girdisi**  
- Uses 10-minute precipitation labels from `data/precip_10min_long_balanced.csv`.  
  (`data/precip_10min_long_balanced.csv` içindeki 10 dakikalık yağış etiketlerini kullanır.)
- Demo station used in the script: `NMS0101`.  
  (Script içinde kullanılan demo istasyon: `NMS0101`.)

3) **Time matching**  
(3) **Zaman eşleme**  
- Parses timestamps from radar filenames (the `-YYYYMMDDHHMMSS` part).  
  (Radar dosya adlarından zaman damgasını (`-YYYYMMDDHHMMSS` kısmı) parse eder.)
- For each precipitation timestamp, finds the nearest radar file within **±10 minutes**.  
  (Her yağış zaman damgası için **±10 dakika** içinde en yakın radar dosyasını bulur.)

4) **Feature extraction (very simple baseline)**  
(4) **Özellik çıkarma (çok basit baseline)**  
- Reads DBZH reflectivity from the HDF5 structure and converts to dBZ using gain/offset metadata.  
  (HDF5 yapısından DBZH reflektiviteyi okur ve gain/offset metadata ile dBZ’e çevirir.)
- Extracts only three global features from the 2D frame:  
  (2B frame’den sadece üç global özellik çıkarır:)
  - `dbz_mean`  
    (`dbz_mean`)
  - `dbz_max`  
    (`dbz_max`)
  - `dbz_p90`  
    (`dbz_p90`)

5) **Model**  
(5) **Model**  
- Baseline: `LinearRegression()` with a simple train/test split.  
  (Baseline: Basit train/test split ile `LinearRegression()`.)

6) **Outputs**  
(6) **Çıktılar**  
- Saves figures and a run log into `outputs/`.  
  (Figürleri ve çalıştırma logunu `outputs/` içine kaydeder.)

---

## How to run
## Nasıl çalıştırılır

Create and activate your environment (example).  
(Ortamını oluşturup aktive et (örnek).)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt