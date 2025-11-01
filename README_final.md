# 🚀 ING Datathon 2025 — Müşteri Kaybı (Churn) Tahmini

Bu depo, **ING Datathon** kapsamında müşterilerin **referans tarihini takip eden 6 aylık dönemde** churn (müşteri kaybı) olasılıklarını tahmin etmek için geliştirdiğim çözümü içerir.

- **Problem:** İkili sınıflandırma (Binary Classification) — churn riski tahmini  
- **Yaklaşım:** Güçlü **özellik mühendisliği** + **LightGBM** + **Optuna** ile hiperparametre optimizasyonu  
- **Felsefe:** Veri setini domain bilgisiyle zenginleştirmek, model mimarisinden daha önceliklidir.

---

## ✨ Amaç ve Metodoloji

Amaç, bankanın müşteri verileriyle **yüksek isabetli** bir churn sınıflandırma modeli kurmaktır.  
Temel strateji:

1. **Özellik Mühendisliği (Feature Engineering):**  
   Ham veriden anlamlı davranışsal sinyaller üretmek (özellikle **aktivite değişim oranları** ve **tenure–yaş etkileşimleri**).
2. **Modelleme:**  
   Kategorik değişkenleri verimli işleyebilen ve pratikte güçlü performans veren **LightGBM**.
3. **Optimizasyon:**  
   **Optuna** ile kritik hiperparametrelerin (learning_rate, num_leaves, max_depth, subsample, colsample_bytree, reg_alpha, reg_lambda, scale_pos_weight) ayarlanması.
4. **Doğrulama:**  
   **5-Fold Stratified K-Fold** + OOF (Out-of-Fold) değerlendirme.
5. **Metrik:**  
   Yarışmanın özel birleşik metriği (**Gini**, **Recall@10%**, **Lift@10%**; ağırlıklar: 0.4 / 0.3 / 0.3), baseline’a oranlanarak hesaplanır.

> Not: Kod çıktılarında benim OOF skorum **~1.01** civarındadır. Optuna ile denemeler sonunda daha yüksek bir OOF değeriniz varsa onu buraya yazabilirsiniz (örn. `1.025165`).

---

## 🔑 Anahtar Sonuçlar

- **OOF Final Skoru (örnek):** `1.014937`  
- **Model:** LightGBM (GBDT)  
- **Doğrulama:** 5-Fold Stratified K-Fold (OOF tahminleriyle)  
- **Öne Çıkan Nokta:** Aktivite trendleri + tenure normalizasyonu gibi **davranışsal sinyaller**

---

## 🛠️ Teknik Yaklaşım

### 1) Veri Hazırlama ve Ön İşleme
- **Birleştirme:** `df_customers`, `df_history`, `df_ref_train`, `df_ref_test` → **df_master**
- **Eksik Değer Yönetimi:**
  - İşlem/miktar sütunlarındaki NaN’ler → **0**
  - `work_sector` eksikleri → **"MISSING"**
- **Zaman Penceresi:** Referans tarihinden **geriye 180 gün (6 ay)** → **df_history_6m**

### 2) İleri Özellik Mühendisliği
**Temel Aktivite Metrikleri (RFM benzeri):**
- Son 6 ay toplam/ortalama EFT & CC sayısı/tutarı:  
  `mobile_eft_cnt_6m_sum`, `mobile_eft_cnt_6m_mean`,  
  `mobile_eft_amt_6m_sum`, `mobile_eft_amt_6m_mean`,  
  `cc_cnt_6m_sum`, `cc_cnt_6m_mean`,  
  `cc_amt_6m_sum`, `cc_amt_6m_mean`
- **Maksimum aktif ürün kategorisi:** `max_prod_cat_6m`  
- **Aktivite yoğunluğu:**  
  `num_unique_months`, `cc_active_months`, `eft_active_months`  
- **Kredi Kartı Aktiflik Oranı:** `cc_activity_ratio = cc_active_months / 6`

**Aktivite Değişim Oranları (Churn Sinyali):**
- `activity_change_ratio = last_month_total_cnt / prev_5_month_avg_cnt`
- `activity_change_ratio_v2 = last_2_months_total_cnt / prev_4_month_avg_cnt`

**Tenure & Demografik Etkileşimler:**
- `age_at_account_open = age - tenure/30.4375`
- `tenure_vs_group_median`, `is_tenure_above_median`
- `avg_eft_cnt_per_month`, `avg_cc_cnt_per_month`
- `religion_tenure_mean`, `work_type_cc_amt_mean`
- `tenure_squared`, `age_squared`, `tenure_age_interaction`
- `age_group`, `work_segment`

### 3) Modelleme ve Optimizasyon
- **Model:** LightGBM (GBDT)
- **Doğrulama:** 5-Fold Stratified K-Fold + OOF
- **Metrik:** `ing_hubs_datathon_metric(y_true, y_prob)`
  - **Gini (0.4)** — `Gini = 2*AUC - 1`  
  - **Recall@10% (0.3)**  
  - **Lift@10% (0.3)**
- **Optuna:** Hiperparametre araması (maksimizasyon)  

---

## 📊 Önemli Özellikler (Feature Importance)

1. `max_prod_cat_6m` — En güçlü sinyal  
2. `mobile_eft_cnt_6m_sum`  
3. `cc_cnt_6m_sum`  
4. `work_segment`  
5. `mobile_eft_amt_6m_sum`  
6. `activity_change_ratio_v2`  
7. `max_cc_amt`  
8. `cc_cnt_6m_mean`  
9. `mobile_eft_cnt_6m_mean`  
10. `avg_eft_cnt_per_month`  
11. `cc_amt_6m_sum`  
12. `avg_cc_cnt_per_month`  
13. `age_at_account_open`  
14. `tenure_vs_group_median`  
15. `last_months_total_cnt`  
16. `tenure`  
17. `tenure_age_interaction`  
18. `activity_change_ratio`  
19. `prev_4_month_avg_cnt`  
20. `age`

---

## 📦 Kurulum ve Çalıştırma

```bash
pip install -U pandas numpy scikit-learn lightgbm optuna matplotlib seaborn
```

Veri dizini:
```python
PATH = "/content/sample_data/Datathon/"
```

### Adımlar:
1. Veri okuma ve temizleme  
2. Özellik mühendisliği  
3. 5-Fold LightGBM eğitimi  
4. (Opsiyonel) Optuna optimizasyonu  
5. `submission.csv` oluşturma

```python
df_test_submission = df_master[df_master['is_train'] == 0][['cust_id']].copy()
df_test_submission['churn'] = test_preds
submission_df = df_test_submission[['cust_id', 'churn']]
# submission_df.to_csv('submission.csv', index=False)
```

---

## 🧩 İpuçları

- **Pandas chained assignment uyarısı:**  
  `df[col] = df[col].fillna(...)` yapısını kullanın.
- **GitHub notebook hatası:**  
  Bu depoda `*_github*.ipynb` dosyaları `metadata.widgets` kaldırılmış ve çıktılar temizlenmiştir.

---

## 📄 Lisans

Kodlar eğitim/araştırma amaçlıdır.  
Katkılar PR ile memnuniyetle kabul edilir.  
**Lisans:** MIT
