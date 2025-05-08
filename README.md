# fall_detection
Real-time fall detection system using YOLOv8 Pose, analyzing human keypoints to identify and alert on potential fall events in video streams.

# 跌倒偵測系統 (Fall Detection with YOLOv8 Pose)

> 以 **Ultralytics YOLOv8 Pose** 搭配自訂關鍵點規則，實現影片中人員跌倒事件的即時偵測，並於畫面標示警示框與提示文字。

---

## 目錄

1. [功能特色](#功能特色)
2. [環境需求](#環境需求)
3. [快速開始](#快速開始)
4. [程式說明](#程式說明)
5. [專案結構](#專案結構)
6. [常見問題](#常見問題)
7. [改進方向](#改進方向)
8. [參考文獻與致謝](#參考文獻與致謝)

---

## 功能特色

* **即時跌倒判斷**：利用 YOLOv8n‑pose 模型取得 17 個人體關鍵點，透過肩膀、腰、腳踝相對高度與姿態長寬比 (H − W) 判斷跌倒。
* **遮擋容錯**：`fall_detection_2()` 函式會過濾低信度 (< 0.5) 關鍵點，降低部分遮擋造成的誤判。
* **多影片格式支援**：可讀取本地影片或 USB / IP 攝影機串流。
* **輸出成品保存**：將偵測後的畫面輸出為 `output_*.mp4`，方便後續審核或報告。

---

## 環境需求

| 軟體            | 版本               | 說明                   |
| ------------- | ---------------- | -------------------- |
| Python        | ≥ 3.9            | 已於 3.11 測試通過         |
| ultralytics   | ≥ 8.2            | YOLOv8 模型與推論 API     |
| opencv‑python | ≥ 4.8            | 影像讀取、顯示與寫入           |
| numpy、torch   | 與 ultralytics 相容 | GPU 推論建議安裝對應 CUDA 版本 |

> **一次安裝**
>
> ```bash
> pip install ultralytics opencv-python
> ```
>
> 若需 GPU/CUDA，請先依官方指引安裝符合版本的 **PyTorch**。

---

## 快速開始

```bash
# 1. 下載/複製專案
$ git clone <repo-url> fall-detection-yolov8
$ cd fall-detection-yolov8

# 2. 安裝相依套件 (建議使用虛擬環境)
$ pip install -r requirements.txt  # 若無 requirements.txt 請參考上方指令

# 3. 準備輸入檔案
#   將欲測試的影片放到 ./videos/test2.mp4，或於執行時指定 --input_path

# 4. 執行範例腳本
$ python fall_detection.py \
    --model yolov8n-pose.pt \
    --input_path videos/test2.mp4 \
    --output_path output/fall_output.mp4 \
    --conf 0.25
```

> **備註**：目前 `fall_detection.py` 內部以硬編碼設定 `video_path`、`output_path`，您可依需求修改，或使用 [程式說明](#程式說明) 節提供的 CLI 版本。

---

## 程式說明

| 函式/類別                | 位置                  | 作用                                      |
| -------------------- | ------------------- | --------------------------------------- |
| `fall_detection()`   | `fall_detection.py` | 基本跌倒判斷，不處理遮擋點                           |
| `fall_detection_2()` | `fall_detection.py` | 進階版本，過濾低信度關鍵點，並於缺失超過 5 點時跳過判斷           |
| `falling_alarm()`    | `fall_detection.py` | 在偵測到跌倒時，繪製紅框與文字提醒                       |
| `__main__` 區段        | `fall_detection.py` | 讀取影片 → 執行 YOLOv8 推論 → 呼叫跌倒判斷 → 視覺化 → 儲存 |

### 跌倒規則 (版本 2)

1. **關鍵點高度關係**

   * 左/右肩 `y` > 腳踝 `y` − `len_factor`
   * 腰 `y` > 腳踝 `y` − (`len_factor` / 2)
   * 肩 `y` > 腰 `y` − (`len_factor` / 2)
2. **姿態扁平化**：`difference = bbox_h - bbox_w`，若 `difference < 0` 表示寬度大於高度，疑似躺倒狀態。
3. 任一側滿足 (1) 或 (2) 即判定為跌倒。

> `len_factor` 為肩膀與腰之間的歐氏距離，用以自適應人體比例。

---

## 專案結構

```
fall-detection-yolov8/
├─ fall_detection.py        # 主程式
├─ requirements.txt         # 相依套件 (選填)
├─ videos/
│  └─ test2.mp4             # 測試影片 (自行放置)
├─ output/
│  └─ fall_output.mp4       # 偵測結果 (自動產生)
└─ README.md                # 專案說明文件
```

---

## 常見問題

| 問題         | 解決方案                                                      |
| ---------- | --------------------------------------------------------- |
| **FPS 偏低** | 確認是否使用 GPU 推論；可改用 `yolov8s/m/l-pose.pt` 取得更高準確度或依硬體選擇較小模型 |
| **模型下載失敗** | 於第一次執行時 Ultralytics 會自動下載模型；亦可手動下載後指定 `--model` 參數指向本地檔案  |
| **視窗未顯示**  | 遠端執行時請刪除 `cv2.imshow()` 或加上 `--headless` 旗標 (需自行擴充)       |

---

## 改進方向

* **CLI 參數化**：將影片路徑、輸出路徑、模型大小、信度閾值等改為命令列參數。
* **Email / Line 通知**：結合即時推播服務，在偵測到跌倒事件時發送通知。
* **深度學習判斷**：以序列 LSTM 或 Transformer 解析關鍵點時間序列，提高準確率並降低誤報。
* **多人體追蹤**：加入 ID 維護與跨影格關聯，支援同時偵測多名被監護者。

---

## 參考文獻與致謝

* **Ultralytics YOLOv8** – [https://docs.ultralytics.com](https://docs.ultralytics.com)
* **COCO Keypoints** Dataset spec
* 感謝原作者提供的 [fall\_detection.py](fall_detection.py) 範例程式。

---

> © 2025 – MIT License. 本文件可自由複製、修改與散布，僅需保留原作者與版權聲明。
