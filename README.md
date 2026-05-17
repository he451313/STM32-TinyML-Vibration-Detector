# 基于 TinyML 的馬達震動異常偵測系統

本專案實現了一個完整的邊緣人工智慧（Edge AI / TinyML）端到端系統，部署於 STM32F446RE 微控制器上。系統透過 I2C 介面即時採集 MPU6500 六軸感測器的三軸加速度訊號，利用滑動視窗（Sliding Window）擷取時序特徵，並運行經過量化壓縮的自編碼器（Autoencoder）神經網路模型。藉由計算輸入訊號與重建訊號之間的均方誤差（MSE）作為異常分數，系統能即時判斷馬達運轉狀態（NORMAL / ABNORMAL），並將狀態與異常分數即時顯示於 SSD1306 OLED 螢幕上。

---

## 1. 系統架構與硬體元件

### 硬體元件
* **微控制器（MCU）**：STMicroelectronics Nucleo-F446RE（搭載 ARM Cortex-M4 核心，具備硬體浮點運算單元 FPU 及 DSP 指令集，時脈 180 MHz，內建 512 KB Flash、128 KB SRAM）。
* **感測器**：MPU6500 六軸慣性測量單元（IMU），本專案使用其三軸加速度計功能，配置為預設量程，透過 I2C1 匯流排進行通訊。
* **顯示模組**：0.96 吋 SSD1306 OLED 螢幕（解析度 128x64），與 MPU6500 共用 I2C1 匯流排，硬體位址為 0x3C（經左移後為 0x78）。

### 系統通訊拓撲
```text
[ STM32F446RE (Master) ]
       │
       ├── I2C1 Bus ─────────────────┬──────────────────┐
       │                             │                  │
       │                      [ MPU6500 (Slave) ] [ SSD1306 OLED (Slave) ]
       │                      位址: 0x68 << 1      位址: 0x3C << 1
       │
       └── UART2 (115200 bps) ──> [ PC 終端機 ] (資料採集與除錯輸出)
```
---

## 2. 軟體環境與第三方函式庫

### 邊緣端開發環境（Embedded C/C++）
* **工具鏈**：VS Code 搭配 CMake 建置系統與 GCC ARM Embedded Toolchain (arm-none-eabi-gcc)。
* **配置與底層驅動**：STM32CubeMX（用於硬體腳位、時脈樹及外設硬體初始化代碼生成）。
* **AI 推論引擎**：X-CUBE-AI v10.2.0（ST 官方最佳化微型神經網路推論運行時庫，自動對接 ARM CMSIS-NN 硬體加速庫）。
* **OLED 驅動庫**：基於 STM32 HAL 封裝的開源 `stm32-ssd1306` 函式庫（包含字體支援庫 `ssd1306_fonts`）。

### 主機端開發環境（Python Data Science）
* **語言版本**：Python 3
* **關鍵資料庫**：
    * `TensorFlow / Keras`：用於神經網路模型的設計、訓練及 TFLite 格式匯出。
    * `Pandas` & `NumPy`：用於資料清洗、時序滑動視窗切片與數學矩陣運算。
    * `Matplotlib` & `Scikit-learn`：用於數據視覺化分析以及資料 MinMaxScaler 正規化處理。

---

## 3. 專案開發流程詳細說明

### 階段一與階段二：硬體配置、序列化與時序數據採集
1.  **硬體配置**：於 STM32CubeMX 中啟用 I2C1（標準模式 100kHz）與 USART2（115200 bps）。
2.  **重定向 printf**：在 `main.c` 中重寫 `_write` 函式，將標準輸出重定向至 `HAL_UART_Transmit`，實現透過序列埠向主機發送數據。
3.  **數據採集**：編寫 Python 腳本 `data_logger.py`，透過串列埠以 50 毫秒（20 Hz）為採樣間隔，分別錄製馬達平穩運轉時的「正常數據（normal_vibration.csv）」與模擬撞擊、軸承卡頓等狀態下的「異常數據（abnormal_vibration.csv）」，每份數據各採集 500 筆樣本。

### 階段三與階段四：模型架構設計、訓練、量化與評估
1.  **時序前處理**：
    * 使用 `MinMaxScaler` 計算正常數據的各軸極值，將原始的 int16 數據精準縮放至 0 到 1 之間。
    * 採用滑動視窗（Sliding Window）機制切割資料，設定視窗大小（Window Size）為 50 筆樣本（代表約 2.5 秒的震動特徵），步長（Step Size）為 10 筆樣本，將資料重塑為形狀如 (樣本數, 50, 3) 的三維張量。
2.  **自編碼器（Autoencoder）架構設計**：
    * 為了適應邊緣端受限的記憶體資源，採用全連接層構成的輕量化自編碼器架構。
    * **輸入層**：將 (50, 3) 的時序視窗壓平成 150 維向量。
    * **編碼器（Encoder）**：Dense(64, ReLU) -> Dense(16, ReLU)。將維度壓縮至 16 維的瓶頸層（Bottleneck），強迫模型擷取核心特徵。
    * **解碼器（Decoder）**：Dense(64, ReLU) -> Dense(150, Sigmoid) -> Reshape(50, 3)。嘗試完美重建輸入訊號。
3.  **模型訓練與量化**：
    * 模型**僅使用正常數據**進行訓練，以均方誤差（MSE）作為損失函式。經過 50 個 Epoch 訓練，模型成功收斂。
    * 使用 `TFLiteConverter` 將 Keras 模型轉換為 TensorFlow Lite 格式，並啟用 `Optimize.DEFAULT` 進行輕量化權重量化，將模型大小壓縮至 29.81 KB。
4.  **閾值（Threshold）評估**：
    * 將正常與異常數據分別餵入模型進行推論，計算重建訊號與原始輸入之間的 MSE 分數。
    * 經視覺化散佈圖分析，正常運轉下的最高 MSE 約為 0.09，而異常撞擊時的 MSE 則會飆升至 0.3 到 1.5 之間。最終選定 **0.25** 作為判斷馬達異常的工業級安全閾值。

### 階段五：邊緣部署、硬體最佳化與綜合除錯
1.  **模型轉換**：將 `model.tflite` 匯入 STM32CubeMX 的 X-CUBE-AI 組件中，選擇 `STM32Cube.AI MCU runtime` 進行代碼生成。分析報告顯示，該模型在編譯後僅需 94.36 KiB 的 Flash 空間與 3.32 KiB 的運行時 RAM，極具邊緣端部署優勢。
2.  **C 語言業務邏輯實作**：在 STM32 的主迴圈中，以 50ms 為週期採集 MPU6500 數據，手動進行 MinMaxScaler 映射後填入滑動視窗緩衝區。當緩衝區集滿 50 筆數據時，呼叫 `ai_network_run` 進行推論，並在 C 語言中手動實現 MSE 計算公式。

---

## 4. 關鍵技術挑戰與解決方案（除錯紀錄）

在落地部署過程中，本專案克服了多項微控制器底層與硬體物理特性的關鍵 Bug：

### 記憶體對齊錯誤（UsageFault / HardFault）
* **技術技術問題**：ARM Cortex-M4 的 DSP 指令集在執行矩陣平行運算時，要求操作數的記憶體位址必須嚴格對齊。若未對齊，會在第二次推論時觸發硬體錯誤。
* **解決方案**：在全域變數宣告中，捨棄舊版手動計算位移的方式，改用新版 X-CUBE-AI 提供的巨集，將神經網路活化區及輸入輸出緩衝區加上 `AI_ALIGNED(32)` 修飾詞，強制進行 32 位元組記憶體對齊。

### 堆疊溢位（Stack Overflow 導致的死機）
* **技術問題技術**：模型執行矩陣乘法時，底層 CMSIS-NN 函式庫會消耗大量的局部變數記憶體。STM32CubeMX 預設的 Stack 空間（1 KB）過小，導致執行第一次推論後便撐爆堆疊，覆蓋系統關鍵變數，引發 HardFault。
* **解決方案**：在 STM32CubeMX 的 Project Manager -> Linker Settings 中，將 Minimum Heap Size 與 Minimum Stack Size 均手動放大至 `0x2000`（8 KB），徹底解決堆疊溢位問題。

### 滑動視窗陣列越界與記憶體損毀
* **技術問題**：在主迴圈內更新時序緩衝區時，若未及時重置計數指標，會導致數組持續累加而發生陣列越界（Array Out of Bounds），寫壞鄰近記憶體。
* **解決方案**：利用標準庫函式 `memmove` 實現高效率的記憶體移動。每次觸發 AI 推論後，將視窗後半部的 40 筆數據（WINDOW_SIZE - STEP_SIZE）整體前移至緩衝區開頭，並將 `sample_count` 精準重置為 40，騰出 10 筆空間給最新採樣，實現無縫的環形滑動視窗。

### 嵌入式 printf 浮點數列印限制
* **技術問題**：為了縮減程式碼體積，STM32 的 GCC 輕量化標準庫（newlib-nano）預設關閉了 `printf` 輸出浮點數（`%f`）的功能，導致除錯時無法在終端機觀察分數變化。
* **解決方案**：在 C 語言端採用整數放大法。將計算出的 `mse_score` 與 `ANOMALY_THRESHOLD` 同時乘以 10000.0f 並強轉為 `int` 型態，以 `分数 / 10000` 的整數形式透過 `%d` 順利輸出，避開了底層限制。

### 高頻震動下的 I2C 通訊當機與死鎖
* **技術問題**：馬達在劇烈震動或遭受物理撞擊時，麵包板上的杜邦線極易產生暫時性接觸不良。這會干擾 I2C 的時脈訊號，導致 STM32 的 I2C 狀態機陷入無限等待的死鎖狀態（Deadlock），導致整顆 MCU 鎖死。
* **解決方案**：在 `HAL_I2C_Mem_Read` 週邊讀取語句中加入錯誤攔截。一旦通訊返回值不為 `HAL_OK`，不允許程式鎖死，而是立即執行暴力救援機制：呼叫 `HAL_I2C_DeInit()` 關閉週邊，延時 10ms 後執行 `HAL_I2C_Init()` 重新初始化 I2C1 暫存器並跳過當前週期，成功實現具備自癒能力的工業級防震韌體。

### OLED 狀態未同步與冷開機重置
* **技術問題**：引入共享 I2C 匯流排的 SSD1306 螢幕後，在反覆進行編譯燒錄（Flash）時，由於 MCU 的 3.3V 電源不中斷，OLED 內部的控制器未經歷硬體重置，常卡在錯誤的通訊時序中，導致序列埠輸出正常但螢幕保持全黑。
* **解決方案**：利用物理冷開機（Hard Reset）機制。在程式燒錄完成後，完全拔除 Nucleo 板的 USB 傳輸線，切斷整體電路供電並等待 3 秒，確保 OLED 晶片電容完全放電。重新上電後，螢幕的 I2C 狀態成功與 MCU 同步，順利顯示即時監控畫面。

---

## 5. 如何運行專案

### 步驟一：硬體連接
1. 將 MPU6500 的 VCC、GND 分別接至 Nucleo-F446RE 的 3.3V 與 GND。
2. 將 MPU6500 的 SCL、SDA 分別接至板上對應的 I2C1_SCL (PB6 或 PB8) 與 I2C1_SDA (PB7 或 PB9)。
3. 將 SSD1306 OLED 的 VCC、GND、SCL、SDA 接線，以並聯方式同樣接入上述相同的電源與 I2C 節點。

### 步驟二：主機端數據驗證與模型匯出
1. 進入 `python/` 資料夾，執行極值分析腳本：
   ```bash
   py find_min_max.py
   ```
   記錄終端機輸出的最大最小值，並將其填入 main.c 中的 MinMaxScaler 參數 區塊。
2. 執行神經網路訓練與 TFLite 轉換腳本：

Bash
py train_model.py

確認生成 model.tflite。

### 步驟三：邊緣端建置與燒錄
打開 STM32CubeMX，載入 .ioc 設定檔，匯入 model.tflite，確認分析無誤後點擊 GENERATE CODE。

使用 VS Code 打開專案。按下 Ctrl + Shift + P 執行 CMake: Delete Cache and Reconfigure。

確認 CMakeLists.txt 中已將 Core/Src/ssd1306.c 與 Core/Src/ssd1306_fonts.c 包含於 target_sources 內。

點擊底部的 Build 進行編譯，編譯通過後點擊 Flash 進行燒錄。

斷開 USB 連接線進行硬體冷開機，隨後重新通電，即可在 OLED 螢幕上實時觀測馬達運轉狀態與 AI 異常分數。
