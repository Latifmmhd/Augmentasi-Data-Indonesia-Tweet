# Dokumentasi Project: Augmentasi Data Deteksi Emosi Bahasa Indonesia (Base Model)

## Ringkasan Project

Project ini bertujuan untuk melakukan **augmentasi data teks** pada dataset deteksi emosi berbahasa Indonesia yang bersumber dari tweet. Augmentasi dilakukan dengan teknik **few-shot completion paraphrasing** menggunakan model bahasa besar **Llama-3-8B versi Base** (bukan Instruct), untuk menyeimbangkan jumlah sample pada tiap kelas emosi hingga mencapai target **2100 sample per label**.

Enam label emosi yang ditangani: `joy`, `sad`, `anger`, `fear`, `love`, `neutral`.

Output akhir berupa dataset gabungan (data asli + data hasil augmentasi) yang disimpan sebagai file CSV baru, siap digunakan untuk melatih model klasifikasi emosi.

---

## 1. Alasan Pemilihan Pendekatan: Base Model vs Instruct Model

Notebook ini secara eksplisit merupakan **revisi** dari versi sebelumnya yang menggunakan model Instruct. Perbedaan mendasar antara base model dan instruct model memengaruhi seluruh desain pipeline:

| Aspek | Model Instruct | Model Base (dipakai di sini) |
|---|---|---|
| Cara kerja | Mengikuti instruksi eksplisit (system/user prompt) | Melanjutkan pola teks (completion) |
| Chat template | Punya `chat_template`, dipakai via `apply_chat_template()` | Umumnya tidak punya `chat_template` — akan error jika dipaksakan |
| Strategi prompting | Instruksi langsung ("tulis ulang tweet ini...") | Few-shot: beberapa contoh pasangan tweet asli → parafrase, diakhiri tweet target |
| Reliabilitas format | Lebih konsisten mengikuti format yang diminta | Kurang reliable, lebih sering menghasilkan output yang gagal validasi |
| Penanganan token akhir | Ada penanganan token `<|eot_id|>` | Tidak relevan, karena bukan bagian chat template base model |

**Alasan pemilihan few-shot completion:** karena base model tidak di-fine-tune untuk mematuhi instruksi, cara paling reliable untuk mengarahkan output adalah dengan memberi **contoh pola** (tweet asli → hasil parafrase) yang konsisten, sehingga model meniru pola tersebut alih-alih "diberi tahu" lewat instruksi yang mungkin diabaikan.

**Konsekuensi desain yang mengikuti pilihan ini:**
1. Prompt dibangun sebagai **string plain-text** biasa, bukan struktur `messages` + `apply_chat_template()`.
2. Ekstraksi hasil generate tetap dilakukan **berbasis token** (bukan string slicing) — ini adalah fix umum yang dipertahankan dari revisi sebelumnya, karena pendekatan `full_output[len(prompt):]` rawan meleset karakter akibat perbedaan representasi string vs token.
3. Fungsi pembersihan (`clean_generated_text`) dipertahankan sebagai jaring pengaman tambahan, karena base model masih berpotensi meniru pola tanda kutip dari data pretraining meskipun sudah dipandu contoh few-shot.
4. Tidak ada penanganan token `<|eot_id|>` karena token tersebut spesifik untuk chat template Instruct. Base model jarang menghasilkan EOS di tengah dokumen sehingga generasi sering berjalan hingga `max_new_tokens` — ini kondisi normal yang ditangani lewat pemotongan baris pertama saja.
5. Disarankan menaikkan `max_retries` dan memperbanyak pengecekan manual (QA) karena base model kurang reliable dibanding Instruct dalam mengikuti gaya/format output.

---

## 2. Alur Pipeline (High-Level)

1. Install & import library
2. Load dataset CSV
3. Tampilkan distribusi label awal
4. Identifikasi kelas yang perlu diaugmentasi
5. Buat prompt few-shot (contoh + target) untuk augmentasi
6. Generate data baru menggunakan Llama-3-8B (Base)
7. Ekstraksi hasil generate berbasis token + pembersihan teks
8. Filter duplicate & validasi kualitas output
9. Gabungkan dataset asli dan hasil augmentasi
10. Tampilkan distribusi label akhir
11. Simpan hasil akhir ke CSV baru

> **Catatan lingkungan:** notebook ini dirancang untuk dijalankan di Google Colab dengan runtime **GPU A100**, mengingat kebutuhan VRAM untuk memuat model 8B parameter (meski sudah dikuantisasi).

---

## 3. Konfigurasi Lingkungan (Environment Setup)

### 3.1 Instalasi Library

```bash
pip install -q transformers
pip install -q accelerate
pip install -q pandas tqdm
pip install -q gdown
```

> **[REVISI]** `bitsandbytes` **dihapus** dari instalasi karena quantisasi 4-bit tidak lagi digunakan. Model kini dimuat langsung dalam presisi BFloat16 penuh (lihat Bagian 3.4 dan Langkah 6).

**Alasan pemilihan tiap library:**

| Library | Fungsi | Alasan Pemilihan |
|---|---|---|
| `transformers` | Memuat model & tokenizer Llama-3 | Library standar HuggingFace untuk load dan menjalankan LLM open-source |
| `accelerate` | Distribusi model ke GPU otomatis | Menangani `device_map="auto"` sehingga model bisa dipetakan otomatis ke device yang tersedia |
| `pandas`, `tqdm` | Manipulasi data & progress bar | Kebutuhan standar untuk memproses dataset CSV dan memantau progres augmentasi |
| `gdown` | Download dataset dari Google Drive | Dataset sumber (`all_data.csv`) disimpan di Google Drive dan diunduh via file ID |

### 3.2 Import Library & Konfigurasi Global

```python
import os, time, random, warnings
import pandas as pd, numpy as np, torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
```

- `warnings` dan `logging` transformers di-suppress ke level `ERROR` agar output notebook tetap bersih dari peringatan non-kritis.
- **Seed global** diset ke `SEED = 42` dan diterapkan ke `random`, `numpy`, dan `torch` (termasuk `torch.cuda.manual_seed_all`) untuk menjaga **reproducibility** hasil antar-run, meskipun proses generation tetap menggunakan sampling stokastik (`do_sample=True`).

### 3.3 Sumber Dataset

Dataset diunduh dari Google Drive menggunakan `gdown` dengan file ID tertentu, disimpan sebagai `all_data.csv`. Pendekatan ini dipilih karena dataset sumber kemungkinan berukuran cukup besar dan disimpan di penyimpanan cloud pribadi, bukan repository publik.

### 3.4 [REVISI] Presisi Model: BFloat16 Penuh (Tanpa Quantisasi)

Versi ini menghapus quantisasi 4-bit (`BitsAndBytesConfig` dari `bitsandbytes`) yang digunakan pada revisi sebelumnya. Model kini dimuat langsung dalam presisi **BFloat16 penuh** melalui parameter `torch_dtype=torch.bfloat16` pada `AutoModelForCausalLM.from_pretrained()`, tanpa `quantization_config` sama sekali.

**Alasan/implikasi perubahan ini:**
- **Akurasi lebih tinggi** — tanpa kuantisasi, bobot model tidak mengalami kompresi presisi (4-bit → 16-bit), sehingga hasil generation berpotensi lebih stabil dan berkualitas dibanding versi terkuantisasi.
- **Kebutuhan VRAM meningkat signifikan** — kira-kira 2x lipat dibanding versi 4-bit (sekitar ~16GB hanya untuk bobot model 8B parameter, belum termasuk aktivasi dan KV-cache saat batch generation). Oleh karena itu, GPU dengan VRAM besar (mis. **A100 40GB/80GB**) menjadi wajib, bukan sekadar disarankan.
- **`bitsandbytes` tidak lagi menjadi dependensi** — dihapus dari daftar instalasi karena tidak dipakai.
- **`model.eval()` kini aman dipanggil** — pada versi 4-bit sebelumnya, pemanggilan `.eval()` atau `.to()` pada model hasil kuantisasi `bitsandbytes` memicu `ValueError`. Karena model sekarang dimuat dalam presisi penuh (tanpa lapisan quantized `Linear8bit`/`Linear4bit`), `model.eval()` dapat dipanggil secara normal untuk menonaktifkan layer seperti dropout selama inferensi.
- **`device_map="auto"` tetap dipertahankan** — distribusi otomatis model ke device (GPU/CPU) via `accelerate` tetap relevan terlepas dari presisi yang digunakan.

---

## 4. Konfigurasi Utama (`CONFIG`) dan Alasan Pemilihan Parameter

```python
CONFIG = {
    "input_csv_path": "all_data.csv",
    "output_csv_path": "all_data_augmented.csv",
    "model_name": "meta-llama/Meta-Llama-3-8B",
    "target_per_label": 2100,
    "emotion_labels": ["joy", "sad", "anger", "fear", "love", "neutral"],
    "batch_size": 32,
    "max_retries": 3,
    "generation_config": {
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "max_new_tokens": 128,
    },
}
```

**Penjelasan dan alasan tiap parameter:**

- **`model_name = "meta-llama/Meta-Llama-3-8B"`** — dipilih model *base* (bukan `-Instruct`) sesuai tujuan revisi notebook ini, yaitu menguji pendekatan few-shot completion tanpa fine-tuning instruksi.
- **`target_per_label = 2100`** — target jumlah sample per label agar dataset menjadi **seimbang (balanced)** antar kelas, sehingga model klasifikasi yang dilatih nantinya tidak bias terhadap kelas mayoritas.
- **`batch_size = 32`** — jumlah prompt yang digenerate sekaligus dalam satu batch, disesuaikan dengan kapasitas VRAM GPU (A100). Batching mempercepat proses generation dibanding memproses satu-per-satu.
- **`max_retries = 3`** — karena base model cenderung lebih sering menghasilkan output yang gagal validasi dibanding Instruct (kurang konsisten mengikuti format), jumlah percobaan per batch dilebihkan (`current_batch = min(batch_size, remaining * max_retries)`) agar target tetap tercapai secara efisien. Nilai ini disarankan dinaikkan (4–5) bila proses augmentasi terasa lambat mencapai target.
- **`temperature = 0.7`** — mengontrol tingkat kreativitas/variasi output. Nilai 0.7 dipilih sebagai titik tengah: cukup tinggi untuk menghasilkan variasi kalimat yang natural (bukan pengulangan kata demi kata dari sumber), namun tidak terlalu tinggi hingga menghasilkan teks yang tidak koheren.
- **`top_p = 0.9`** — nucleus sampling, mengambil token dari kumpulan token dengan probabilitas kumulatif 90% teratas. Dikombinasikan dengan temperature untuk membatasi token-token dengan probabilitas sangat rendah (noise) tanpa membuat output terlalu deterministik.
- **`do_sample = True`** — wajib diaktifkan agar `temperature` dan `top_p` berfungsi (jika `False`, model akan menggunakan greedy/deterministic decoding).
- **`max_new_tokens = 128`** — dibatasi karena hasil augmentasi adalah tweet pendek (1 kalimat), sehingga 128 token dianggap cukup lebih dari batas atas panjang natural sebuah tweet, sekaligus membatasi waktu komputasi per generasi.

---

## 5. Tahapan Proses Secara Berurutan

### Langkah 1 — Install Library
Menginstal seluruh dependensi (`transformers`, `bitsandbytes`, `accelerate`, `pandas`, `tqdm`, `gdown`) seperti dijelaskan di Bagian 3.1, kemudian mengunduh dataset sumber dari Google Drive.

### Langkah 2 — Import Library & Konfigurasi Global
Mengimpor seluruh library yang dibutuhkan, menonaktifkan warning yang tidak relevan, menetapkan seed global untuk reproducibility, dan mendefinisikan dictionary `CONFIG` yang menjadi pusat pengaturan seluruh pipeline (lihat Bagian 4).

### Langkah 3 — Load Dataset CSV
Fungsi `load_dataset()` bertugas:
- Membaca file CSV menggunakan `pandas`.
- Memvalidasi keberadaan kolom wajib (`tweet`, `label`); jika hilang, akan melempar `ValueError`.
- Membuang baris dengan nilai kosong pada `tweet` atau `label`.
- Merapikan tipe data kolom `tweet` menjadi string dan membuang whitespace berlebih.
- Menormalisasi kolom `label` ke huruf kecil.
- Membuat kolom `index` dari index DataFrame apabila belum tersedia di CSV, agar setiap baris memiliki identitas unik yang bisa dijadikan referensi saat penggabungan data nanti.

### Langkah 4 & 5 — Distribusi Label Awal & Identifikasi Kebutuhan Augmentasi
- `analyze_distribution()` menghitung jumlah sample per label menggunakan `value_counts()`, menampilkannya dalam bentuk teks maupun **bar chart** (matplotlib) dengan palet warna berbeda untuk tiap label, sehingga ketimpangan distribusi kelas dapat langsung terlihat secara visual.
- `identify_augmentation_needs()` membandingkan jumlah sample tiap label terhadap `target_per_label` (2100). Selisihnya (`target - current`, minimal 0) menjadi jumlah sample baru yang perlu dibuat untuk label tersebut. Label yang sudah mencapai atau melebihi target akan dilewati (tidak diaugmentasi lebih lanjut), untuk menghindari over-generation yang tidak perlu.

### Langkah 6 — Load Model Llama-3-8B (Base, BFloat16 Penuh — Tanpa Quantisasi)

**Pengecekan GPU:** notebook terlebih dulu memverifikasi ketersediaan GPU, nama GPU, kapasitas VRAM, dan dukungan BFloat16 — informasi ini penting karena proses loading dan inferensi model 8B parameter dalam presisi penuh sangat bergantung pada spesifikasi hardware (VRAM harus lebih besar dibanding kebutuhan versi terkuantisasi).

**Autentikasi HuggingFace:** menggunakan `huggingface_hub.login()` karena model Llama-3 bersifat *gated* (memerlukan akses/lisensi dari Meta melalui akun HuggingFace).

**[REVISI] Loading model tanpa quantisasi:**
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # BF16 penuh untuk bobot & komputasi
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
```
- `BitsAndBytesConfig` dan parameter `quantization_config` **dihapus sepenuhnya**. Tidak ada lagi kuantisasi 4-bit (NF4) maupun double quantization.
- `torch_dtype=torch.bfloat16` — bobot model dimuat dan komputasi forward pass dilakukan langsung dalam presisi BFloat16 penuh (16-bit), format yang didukung native dan optimal pada GPU A100, tanpa proses dekompresi kuantisasi saat inferensi.
- `model.eval()` — **kini dipanggil secara eksplisit** untuk menonaktifkan layer seperti dropout selama inferensi. Ini berbeda dari versi sebelumnya, di mana pemanggilan `.eval()`/`.to()` pada model hasil kuantisasi `bitsandbytes` memicu `ValueError`; karena model sekarang tidak lagi memiliki layer quantized, pemanggilan ini aman dan menjadi praktik standar.

**Konfigurasi tokenizer** (tidak berubah):
- `padding_side="left"` — krusial untuk *batch generation* pada model *causal LM* seperti Llama, karena padding di kiri memastikan token terakhir dari setiap sequence dalam batch berada pada posisi yang sama, sehingga proses generate berikutnya berjalan konsisten antar baris dalam batch. Konfigurasi ini juga menjadi prasyarat agar ekstraksi hasil berbasis `prompt_len` (lihat Langkah 8) dapat akurat.
- Karena Llama-3 secara default tidak memiliki `pad_token`, tokenizer diset agar `pad_token` mengacu ke `eos_token`.

**Distribusi model ke device:**
- `device_map="auto"` — tetap memanfaatkan `accelerate` untuk mendistribusikan layer model secara otomatis ke device yang tersedia (GPU/CPU), terlepas dari presisi yang digunakan.
- Device utama model dideteksi melalui `model.hf_device_map` untuk memastikan input tensor dipindahkan ke device yang tepat saat inferensi, terutama penting jika model tersebar di beberapa device (multi-GPU/CPU-offload).

> **Dampak kebutuhan hardware:** dengan presisi BFloat16 penuh, kebutuhan VRAM untuk bobot model saja meningkat menjadi kira-kira **~16GB** (2x lipat dibanding versi 4-bit sebelumnya), belum termasuk memori untuk aktivasi dan KV-cache saat batch generation (`batch_size=32`). GPU dengan VRAM besar seperti **A100 40GB/80GB** menjadi persyaratan wajib, bukan sekadar rekomendasi.

### Langkah 7 — Prompt Template Augmentasi (Few-Shot Completion)

**`EMOTION_DESCRIPTIONS`** — dictionary yang memetakan tiap label emosi ke deskripsi naratif dalam Bahasa Indonesia (misalnya `joy` → "kebahagiaan, kesenangan, kegembiraan, atau antusiasme"). Deskripsi ini disisipkan ke dalam prompt untuk memperkaya konteks semantik, membantu model memahami nuansa emosi yang harus dipertahankan saat parafrase.

**`FEW_SHOT_EXAMPLES`** — tiga contoh pasangan (tweet asli → hasil parafrase) yang mewakili label `joy`, `sad`, dan `anger`. Contoh-contoh ini sengaja ditulis singkat, bergaya Twitter natural (bahasa gaul/slang), tanpa tanda kutip, agar base model dapat meniru format tersebut secara konsisten melalui pattern-matching, bukan melalui pemahaman instruksi eksplisit.

**Fungsi `build_augmentation_prompt(original_text, label)`:**
Membangun prompt final dengan struktur:
1. Kalimat pembuka singkat yang menjelaskan konteks (bukan instruksi ketat, melainkan narasi pengantar pola).
2. Blok few-shot examples, masing-masing berformat `Tweet asli / Emosi / Hasil parafrase`.
3. Baris target: tweet asli yang ingin diparafrase, label emosinya, dan diakhiri dengan `Hasil parafrase:` (tanpa isi) — memancing model untuk melanjutkan (complete) baris tersebut sesuai pola yang telah dicontohkan.

Pendekatan ini memastikan label emosi tetap menjadi bagian dari **pola teks** (bukan instruksi terpisah), sehingga model lebih mudah mengasosiasikan label dengan gaya bahasa yang sesuai.

### Langkah 8 — Fungsi Validasi & Filter Duplikat

Tiga fungsi utama pada tahap ini:

**1. `clean_generated_text(text)`** — jaring pengaman pembersihan output mentah:
- Mengambil hanya **baris pertama non-kosong** dari hasil generate (karena base model sering melanjutkan generasi melebihi satu kalimat/tweet).
- Membuang tanda kutip ganda yang nyelip di mana pun dalam kalimat.
- Membuang tanda kutip tunggal hanya jika membungkus keseluruhan teks (agar tidak merusak apostrof wajar dalam slang).
- Merapikan spasi ganda menjadi satu spasi.

**2. `is_valid_output(generated, original, existing_texts)`** — menerapkan empat kriteria validasi kualitas secara berurutan:
- **Panjang minimum:** minimal 5 kata, untuk menyaring output yang terlalu pendek/tidak bermakna.
- **Anti-duplikat exact-match:** hasil tidak boleh identik dengan teks yang sudah ada di `existing_texts` (set gabungan data asli + data augmentasi yang sudah dihasilkan sejauh ini).
- **Anti-plagiat via Jaccard similarity:** dihitung sebagai rasio irisan terhadap gabungan token antara teks asli dan hasil generate; jika similarity **> 0.85**, hasil ditolak karena dianggap terlalu mirip (bukan parafrase yang bermakna, melainkan pengulangan kata demi kata).
- **Bebas artefak prompt:** menyaring output yang masih mengandung sisa struktur prompt atau token spesial (misalnya `"tweet asli:"`, `"hasil parafrase:"`, `"<|"`, `"system"`, `"tentu,"`, dsb.), yang mengindikasikan model gagal melakukan completion secara bersih.

**3. `extract_generated_text(output_ids, prompt_len, tokenizer)`** — mengekstrak hasil generate secara **presisi berbasis token**, bukan string slicing:
- Sebelumnya (versi lama, rawan bug): `full_output[len(prompt):]` — pendekatan ini bisa meleset karena panjang string prompt hasil decode tidak selalu sama persis dengan panjang string prompt asli (perbedaan tokenisasi/detokenisasi).
- Sekarang: `output_ids[prompt_len:]` — memotong langsung pada level **token ID**, menggunakan `prompt_len` (panjang token prompt setelah padding, yang seragam untuk seluruh baris dalam satu batch karena `padding_side="left"`). Hasil potongan token baru kemudian di-decode dan dibersihkan lewat `clean_generated_text()`.

### Langkah 9 — Generate Data Augmentasi (Batching)

Fungsi `generate_augmented_samples()` menjalankan loop generasi untuk satu label emosi hingga jumlah sample valid (`num_needed`) terpenuhi:

1. Pada tiap iterasi, dihitung `remaining` (sisa kebutuhan) dan `current_batch = min(batch_size, remaining * max_retries)` — strategi ini melebihkan ukuran batch aktual berdasarkan `max_retries` untuk mengantisipasi tingginya rasio kegagalan validasi pada base model, sambil tetap dibatasi oleh `batch_size` maksimum (kapasitas VRAM).
2. Sumber teks (`selected_sources`) dipilih secara acak (`random.choices`) dari kumpulan tweet asli berlabel sama.
3. Prompt dibangun untuk seluruh baris dalam batch via `build_augmentation_prompt()`.
4. Tokenisasi batch dengan `padding=True`, `truncation=True`, `max_length=768` — batas panjang dinaikkan dari versi sebelumnya karena prompt kini memuat blok few-shot examples yang lebih panjang.
5. `prompt_len` dicatat dari `input_ids.shape[1]` untuk keperluan ekstraksi presisi token pada Langkah 8.
6. Generation dijalankan dalam `torch.inference_mode()` (menonaktifkan gradient tracking demi efisiensi memori/komputasi) menggunakan parameter dari `generation_config` (Bagian 4).
7. Setiap hasil dalam batch diekstrak dan divalidasi; hasil yang lolos ditambahkan ke daftar `augmented` dan ke set `seen_texts` (mencegah duplikat antar-iterasi berikutnya).
8. Progress bar teks sederhana menampilkan persentase kemajuan menuju `num_needed`.
9. `torch.cuda.empty_cache()` dipanggil tiap akhir iterasi untuk membebaskan memori GPU yang tidak terpakai.

Loop utama kemudian menjalankan fungsi ini untuk setiap label yang membutuhkan augmentasi (`num_needed > 0`), melewati label yang jumlahnya sudah cukup. Setiap sample baru diberi `index` unik yang melanjutkan dari `max_existing_index + 1`, memastikan tidak ada bentrok index dengan data asli.

### Langkah 10 — Gabungkan Dataset & Tampilkan Distribusi Akhir

- Hasil augmentasi (`all_augmented_rows`) dikonversi menjadi DataFrame (`df_augmented`), lalu digabungkan dengan data asli (`df_all`) menggunakan `pd.concat()`.
- Dataset gabungan dirapikan ke tiga kolom standar (`index`, `tweet`, `label`), dengan normalisasi ulang whitespace dan huruf kecil pada label.
- Distribusi label akhir ditampilkan kembali menggunakan `analyze_distribution()` untuk memverifikasi bahwa augmentasi berhasil mendekati/mencapai target per label.
- Ringkasan tabel perbandingan **Sebelum vs Sesudah vs Tambahan** per label dicetak, serta beberapa contoh acak hasil augmentasi ditampilkan untuk pengecekan kualitas secara cepat.

### Langkah 11 — Simpan Dataset Hasil Augmentasi ke CSV

- Dataset final disimpan ke `all_data_augmented.csv` menggunakan `to_csv(index=False)`.
- File yang tersimpan dibaca ulang (`df_verify`) sebagai langkah verifikasi bahwa proses penyimpanan berhasil dan jumlah baris sesuai ekspektasi.
- Tersedia opsi tambahan untuk mengunduh file langsung ke komputer lokal via `google.colab.files.download()`, dengan fallback pesan jika notebook tidak dijalankan di lingkungan Google Colab.

---

## 6. Checklist Kualitas Akhir

Pipeline ini secara eksplisit menjamin kriteria kualitas berikut pada dataset hasil akhir:

- [x] Hanya kelas dengan jumlah sample di bawah target (`< 2100`, sesuai `target_per_label`) yang diaugmentasi.
- [x] Tidak ada duplikat exact-match antar sample (baik terhadap data asli maupun sesama data augmentasi).
- [x] Jaccard similarity terhadap teks sumber dijaga di bawah 0.85, untuk memastikan hasil merupakan parafrase yang bermakna, bukan pengulangan.
- [x] Index baru pada data augmentasi tidak bentrok dengan index data asli.
- [x] Emosi dan makna utama teks asli dipertahankan (dijaga lewat desain prompt few-shot berbasis label emosi).
- [x] Output berupa Bahasa Indonesia yang natural (gaya Twitter/slang), difilter dari artefak prompt maupun tanda kutip yang tidak diinginkan.

---

## 7. Ringkasan Struktur Output

| Kolom | Deskripsi |
|---|---|
| `index` | ID unik setiap baris (tidak ada duplikat dengan data asli) |
| `tweet` | Teks tweet, baik asli maupun hasil augmentasi |
| `label` | Label emosi: `joy`, `sad`, `anger`, `fear`, `love`, `neutral` |

---

## 8. Catatan Tambahan & Rekomendasi

- Karena base model kurang reliable dibanding model Instruct dalam mengikuti gaya/format output, disarankan untuk **mengecek lebih banyak sample hasil augmentasi secara manual** sebelum dataset digunakan untuk training model klasifikasi.
- Jika proporsi output yang gagal validasi tinggi (proses augmentasi berjalan lambat menuju target), pertimbangkan untuk menaikkan nilai `max_retries` (misalnya ke 4–5).
- Notebook ini wajib dijalankan pada runtime dengan GPU memadai (direkomendasikan **A100 40GB/80GB** di Google Colab) mengingat model 8B parameter kini dimuat dalam presisi **BFloat16 penuh tanpa quantisasi**, sehingga kebutuhan VRAM jauh lebih besar dibanding versi 4-bit sebelumnya.
- Jika mengalami `CUDA out of memory` pada GPU dengan VRAM lebih kecil (mis. A100 40GB dengan `batch_size=32`), pertimbangkan untuk menurunkan `batch_size` pada `CONFIG`, karena tidak ada lagi ruang hemat memori dari kuantisasi.
