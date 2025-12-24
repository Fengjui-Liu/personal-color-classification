#  Personal Color Classification (四季色彩分類)

基於臉部特徵的個人色彩季節分類系統，使用電腦視覺與機器學習判斷使用者屬於春/夏/秋/冬哪種色彩類型。

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)

## 專案概述

### 問題定義
「四季色彩理論」將人的膚色、髮色、眼睛顏色分為春、夏、秋、冬四種類型，用於指導穿搭配色。傳統判斷方式依賴專業顧問主觀判斷，本專案嘗試用機器學習自動化這個過程。

### 我的角色
- 特徵工程設計
- 多模型實驗與比較分析
- 模型失敗原因分析

### 技術亮點
- 使用 MediaPipe Face Mesh 精準定位臉部關鍵區域
- 設計 24 維手工特徵（HSV + LAB + 對比度特徵）
- 比較 4 種模型並深入分析失敗原因

---

## 方法

### 1. 特徵提取 Pipeline

```
輸入圖片 → MediaPipe 臉部偵測 → 關鍵區域定位 → 色彩空間轉換 → 24 維特徵向量
                                    ↓
                            臉頰 / 嘴唇 / 眼睛 / 額頭
```

### 2. 特徵設計 

| 類別 | 特徵 | 數量 | 說明 |
|------|------|------|------|
| **HSV** | skin/lip/eye 的 H, S, V | 9 | 色相、飽和度、明度 |
| **LAB** | skin/lip/eye 的 L, A, B | 9 | 對膚色冷暖更敏感 |
| **對比度** | warmth, yellowness, contrast 等 | 6 | 捕捉區域間差異 |

### 3. 關鍵程式碼

```python
# 膚色冷暖度計算 (LAB 色彩空間)
skin_warmth = skin_lab[1] - 128      # A 通道：正=暖，負=冷
skin_yellowness = skin_lab[2] - 128  # B 通道：正=黃，負=藍

# 膚色與嘴唇對比
skin_lip_H_diff = abs(skin_hsv[0] - lip_hsv[0])
```

---

## 實驗結果

### 數據集
- **總樣本數**：4,905 張臉部圖片
- **類別分布**：Winter (1,304) / Autumn (1,292) / Spring (1,179) / Summer (1,130)
- **訓練/測試比例**：75% / 25%

### 模型比較

| 模型 | 準確率 | 說明 |
|------|--------|------|
| K-Means | ARI: 0.03 | 無監督，完全失敗 |
| Perceptron | 35.0% | 線性分類，略優於隨機 |
| SVM (RBF) | 42.2% | 非線性，有改善 |
| **Random Forest** | **46.1%** | 最佳表現 |

### Confusion Matrix 分析

```
              Predicted
Actual     Spring  Summer  Autumn  Winter
Spring       156      48      68      23
Summer        89      71      99      23
Autumn        48      29     231      15
Winter        58      32     172      65
```

**關鍵發現**：
- **Autumn** 辨識最準確（可能因為暖色調 + 深色特徵明顯）
- **Winter 常被誤判為 Autumn**：兩者都是深色系，模型難以區分冷暖
- **Summer 常被誤判為 Spring**：數據集膚色偏深，高光反射誤導模型

---

## 失敗分析與學習

### 為什麼 K-Means 完全失敗？

K-Means 基於歐氏距離分群，但：
- Spring 的淺黃色與亮綠色在色彩空間中距離很遠
- Spring 的綠色可能比 Summer 的綠色在數值上更接近

**結論**：四季色彩不是單純的「顏色相似度」問題，而是複雜的非線性規則。

### 數據集偏差問題

數據集以深膚色為主，導致：
1. 深膚色的高光反射 → 被誤判為 Spring（明亮特徵）
2. 冷暖色調在深膚色上差異較小 → Winter/Autumn 混淆

### 未來改進方向

1. **數據層面**：收集更多亞洲膚色樣本，平衡數據集
2. **特徵層面**：加入紋理特徵、考慮光線正規化
3. **模型層面**：嘗試 CNN 直接從圖片學習特徵

---

## 技術棧

- **電腦視覺**：OpenCV, MediaPipe Face Mesh
- **機器學習**：scikit-learn (KMeans, SVM, RandomForest)
- **資料處理**：NumPy, Pandas
- **視覺化**：Matplotlib, Seaborn

---

### 執行分析

```bash
# 在 Google Colab 開啟
jupyter notebook notebooks/analysis.ipynb
```

---


---

## 👤 作者

**劉豐睿 (Ryder Liu)**
- 國立政治大學 資訊管理學系

---

*這是資料分析課程的期末專案，專注於探索機器學習在主觀色彩判斷問題上的應用與限制。*
