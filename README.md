# 量子序列模型於金融預測與交易之論文重現報告

孫士傑（Shih-Chieh Sun）｜國立臺灣大學電信工程學研究所

本報告重現以下兩篇論文，並將重現結果與論文數值逐項比較：

1. J.-H. Chen, M.-K. Hung, Y.-C. Tsai, S. Y.-C. Chen, "Batched Training for QLSTM vs. QFWP: A System-Oriented Approach to EPC-Aware RMSE-DA," *ICASSP 2026*.（以下稱「實驗一」）
2. J.-H. Chen, Y.-C. Huang, Y.-C. Tsai, S. Y.-C. Chen, "Quantum-Enhanced Forecasting for Deep Reinforcement Learning in Algorithmic Trading," *IEEE Globecom Workshops 2025*.（以下稱「實驗二」）

## 目錄

1. [結論](#conclusion)
2. [重現方法](#method)
3. [實驗一：QLSTM 與 QFWP 之比較](#exp1)
4. [實驗二：QLSTM 結合 QA3C 之交易代理](#exp2)
5. [討論：資料前視偏誤](#discussion)
6. [待加強與後續工作](#future)
7. [程式使用方式](#usage)
8. [專案結構](#structure)
9. [參考文獻](#references)

---

<a id="conclusion"></a>
## 1. 結論

**實驗一（QLSTM vs. QFWP）：論文的主要結論可以重現。**
在參數量對齊（QLSTM 32 個、QFWP 33 個）的條件下，QFWP 在六種 batch size 的方向準確率（DA）均高於 QLSTM，batch 4 至 32 的差異達統計顯著（Wilcoxon p ≤ 0.004），與論文一致。QFWP 的 DA 為 66%～71%，與論文的 66%～73% 相差 0.4～3.2 個百分點。論文指出「batch 加大時，整體訓練加速比上限約為 2 倍」，本重現亦得到相同趨勢：QLSTM 在 batch 64 的加速比為 2.31 倍，論文為 2.23 倍。主要差異在於 QLSTM 於小 batch 時的 DA 比論文低 7～8 個百分點。

**實驗二（QLSTM＋QA3C）：模型架構可以重現，交易結果尚未重現。**
QA3C（244 個參數）與古典 A3C（3,332 個參數）的模型規模與論文完全相同，資料切分、特徵、標籤、狀態、動作、獎勵結構與 A3C 超參數也依論文設定。然而：

- QLSTM 預測器的測試準確率為 57%～66%，低於論文的 71.5%；測試集僅 88 筆，統計誤差約 ±5 個百分點。
- 古典 A3C 的總報酬 9.2%、最大回撤約 1%，與論文的 11.4%、0.92% 屬同一量級，但交易次數約為論文的兩倍、勝率高約 26 個百分點，顯示交易行為不同。
- 論文有多項設定未說明，獎勵公式亦有疑似筆誤，是無法完全重現的主因。

**資料面的發現。**
Yahoo Finance 匯率日線的收盤價約等於該 K 棒開始時的價格，但最高價與最低價涵蓋整日，因此同一列的開高低價隱含隔日收盤的資訊。不使用任何模型，僅比較當日高低價中點與收盤價，即可判斷隔日漲跌方向 73.1%；同規模的古典 LSTM 也達到 69% 的 DA。實驗一約 70% 的 DA 可能主要來自此資料特性，量子模型本身的貢獻仍需在修正資料後重新驗證。

---

<a id="method"></a>
## 2. 重現方法

兩個實驗均以 PennyLane `lightning.qubit` 模擬器、adjoint 微分及 PyTorch 實作，與論文使用的工具相同。另以 NumPy 獨立實作 state-vector 模擬器與反向自動微分，作為交叉驗證；兩種實作的輸出與梯度差異在 1e-15 以內（`pl_check.py`）。

| 項目 | 規格 |
|---|---|
| 軟體 | Python 3.12、PennyLane 0.45.1、PyTorch 2.14.1 |
| 硬體 | Apple M 系列 CPU（論文：Intel Core Ultra 7 265K、WSL2） |
| 資料來源 | Yahoo Finance（`EURUSD=X`、`TWD=X`、貨幣 ETF） |
| 統計檢定 | Wilcoxon signed-rank test、Cliff's δ |

論文未說明的設定由本重現自行決定，均列於各實驗的說明中，完整對照見 [`REPORT_design.md`](REPORT_design.md)。

---

<a id="exp1"></a>
## 3. 實驗一：QLSTM 與 QFWP 之比較

### 3.1 實驗設定

| 項目 | 論文 | 本重現 |
|---|---|---|
| 資料 | EUR/USD 日線 OHLC，2003-12～2024-12 | 相同（2003-12-01～2024-12-31） |
| 樣本 | 序列長度 5，預測隔日收盤；依時間 80/20 切分 | 相同；訓練 4,374 筆（與論文相同）、測試 1,094 筆 |
| 正規化 | 每個視窗各自 min–max，目標使用相同尺度 | 相同 |
| QLSTM | H=3、7 qubits、深度 1，共 32 參數 | 相同 |
| QFWP | 3 qubits、D=1、A=1，共 33 參數 | 相同 |
| 訓練 | 2 epochs；batch 為 non-batch、4、8、16、32、64；10 個隨機種子 | 相同 |
| 最佳化器 | 未說明 | Adam，學習率 0.05 |

論文未說明最佳化器與學習率。本重現以 batch 4 與 16、3 個種子掃描 9 組設定，選出 DA 最接近論文者（`sweep1.py`）。QFWP 的 4 維輸入無法直接放入 3 個 qubit，因此先將 encoder 的 3 維輸出以 Ry 編碼。

### 3.2 預測準確率

下表為 10 個種子的平均值 ± 標準差；括號內為論文數值。

| Batch | DA QLSTM (%) | DA QFWP (%) | RMSE QLSTM | RMSE QFWP | Wilcoxon p |
|---|---|---|---|---|---|
| non-batch | 60.33 ± 2.20 | 63.06 ± 6.89 | 0.00421 | 0.00455 | 0.85 |
| 4 | 60.74 ± 1.72（69.04） | 70.11 ± 1.45（72.72） | 0.00424（0.0034） | 0.00345（0.0030） | 0.004 |
| 8 | 60.32 ± 1.50（67.89） | 69.60 ± 4.38（72.75） | 0.00407（0.0036） | 0.00318（0.0029） | 0.002 |
| 16 | 59.64 ± 1.12（65.56） | 70.98 ± 0.83（72.34） | 0.00416（0.0039） | 0.00301（0.0029） | 0.002 |
| 32 | 58.47 ± 1.33（62.34） | 69.50 ± 2.27（70.18） | 0.00418（0.0043） | 0.00313（0.0031） | 0.002 |
| 64 | 57.22 ± 1.82（58.67） | 66.20 ± 4.07（65.84） | 0.00440（0.0049） | 0.00371（0.0034） | 0.084 |

作為參考，隨機漫步模型（以今日收盤預測明日收盤）的 RMSE 為 0.00499、DA 為 50.5%；與 QLSTM 相同 hidden 大小的古典 LSTM，DA 為 68.6%～69.7%。

### 3.3 訓練加速比

加速比定義為 non-batch 訓練時間除以各 batch 的訓練時間（10 個種子的中位數）。

| Batch | QLSTM | 論文 | QFWP | 論文 |
|---|---|---|---|---|
| 4 | 1.74 | 1.41 | 1.91 | 1.37 |
| 8 | 2.02 | 1.67 | 2.26 | 1.63 |
| 16 | 2.18 | 1.90 | 2.46 | 1.77 |
| 32 | 2.32 | 2.08 | 2.52 | 1.94 |
| 64 | 2.31 | 2.23 | 2.59 | 2.02 |

### 3.4 與論文之比較

與論文一致之處有三點。第一，六種 batch size 下 QFWP 的 DA 均高於 QLSTM、RMSE 均較低，batch 4 至 32 的差異達統計顯著。第二，QFWP 的 DA 與論文相差 0.4～3.2 個百分點，RMSE 在 batch 16 與 32 分別為 0.00301 與 0.00313，接近論文的 0.0029 與 0.0031。第三，加速比在 batch 32 以上趨近 2.3 倍後不再增加，與論文「反向傳播的加速有限，限制整體加速比」的觀察一致。

與論文不同之處亦有三點。第一，QLSTM 於 batch 4 與 8 的 DA 比論文低 7～8 個百分點，且幾乎不隨 batch 變化；論文中 QLSTM 的 DA 則隨 batch 增大而明顯下降。推測原因是學習率：本重現的學習率依 QFWP 的結果選定，未必適合 QLSTM。第二，QFWP 在 batch 8 與 64 的種子間變異較大，標準差約 4 個百分點。第三，加速比整體略高於論文，可能與硬體及套件版本不同有關。

---

<a id="exp2"></a>
## 4. 實驗二：QLSTM 結合 QA3C 之交易代理

### 4.1 實驗設定

| 項目 | 論文 | 本重現 |
|---|---|---|
| 資料 | USD/TWD 日線，2000-01-01～2025-04-30 | Yahoo 最早僅至 2004-03-24；扣除均線暖機後自 2004-07-19 起 |
| 切分 | 訓練至 2020-03-31；測試 2020-04-01～2025-04-30 | 相同 |
| QLSTM 預測器 | 最近 4 日 ×（開、高、低、收、MA5、MA10）；5 日漲跌超過 ±1.2% 之二元標籤；hidden 2；50 epochs；RMSprop 5e-3；weighted cross-entropy | 相同（batch 32） |
| 狀態與動作 | 10 維狀態；持有、買進 1 單位、全部賣出；只做多 | 相同 |
| 獎勵 | −0.02 加上進場、出場、持有三項，截斷於 [−15, 30] | 相同（虧損出場項之修正見下） |
| QA3C | 8-qubit VQC；244 參數（量子 32、古典 212） | 相同 |
| 古典 A3C | 3,332 參數 | 相同（actor 10→128→3、critic 10→128→1） |
| A3C 超參數 | 8 個 worker、rollout 30、Shared Adam 學習率 1e-5、γ 0.995、β 0.05 | 相同；8 個非同步行程 |
| 訓練量 | 獎勵曲線約於第 1,400 個 episode 收斂 | 1,600 個 episode |

論文未說明、由本重現自行設定的項目如下：

| 項目 | 本重現之設定 |
|---|---|
| VQC 結構 | \|+⟩ → Ry(x) → [CNOT 環 → Rz] → [CNOT 環 → Ry] → 量測 8 條線；actor 與 critic 各一個 VQC。此為同時符合「量子 32、古典 212」參數量之配置 |
| 初始資金與部位 | 初始資金為 10 單位（期初價格的 10 倍），每次買進 1 單位 |
| 無效動作 | 罰 −0.1 |
| 虧損出場 | 論文表格為 −2 − 10×pnl%，虧損時 pnl% 為負，依原式反而得到正獎勵，判斷為筆誤，改用 −2 − 10×\|pnl%\| |
| Episode | 完整跑過訓練期間一次 |
| 測試時的動作選擇 | 選擇機率最高的動作 |
| 相對波動 | 20 日波動除以過去一年 20 日波動的平均 |
| 錯誤報價 | Yahoo 的 USD/TWD 有明顯錯誤報價（例如 2011-10-26 單日上漲 1571%），與前後兩日平均相差 25% 以上的資料列予以刪除，共 10 列 |

### 4.2 QLSTM 預測器

| | 測試準確率 | 論文 |
|---|---|---|
| PennyLane 實作（3 個種子） | 57.95%、65.91%、56.82%，平均 60.2% | 約 71.5% |
| NumPy 實作（3 個種子） | 64.77%、61.36%、65.91%，平均 64.0% | 約 71.5% |
| 全部預測多數類別 | 51.1% | — |

測試集共 88 筆。論文的 71.5% 約相當於答對 63 筆，本重現為 50～58 筆。

### 4.3 交易代理

測試期間為 2020-04-01 至 2025-04-30。

| | 總報酬 | 最大回撤 | 交易次數 | 勝率 | 持倉時間比例 |
|---|---|---|---|---|---|
| 古典 A3C，PennyLane 實作（3 個種子平均） | 9.16% | 1.06% | 434 | 81.4% | 62% |
| 古典 A3C，NumPy 實作（3 個種子平均） | 9.20% | 0.98% | 435 | 81.2% | 64% |
| QA3C，NumPy 實作（1 個種子） | 4.32% | 0.39% | 69 | 88.4% | 10% |
| QA3C，PennyLane 實作 | 執行中 | | | | |
| 論文：古典 A3C | 11.42% | 0.92% | 230 | 54.78% | — |
| 論文：QA3C | 11.87% | 0.92% | 231 | 56.71% | — |
| 買進並持有美元 | 6.92% | — | — | — | — |

論文以貨幣 ETF 作為比較基準。下表為相同測試期間的實際報酬：

| ETF | 2020-04-01～2025-04-30 實際報酬 | 論文 |
|---|---|---|
| FXB | 14.49% | 13.81% |
| FXF | 13.14% | 12.21% |
| USDU | 11.27% | 4.91% |
| FXE | 5.40% | 4.25% |
| FXC | 7.21% | 1.74% |
| FXA | 7.74% | −5.79% |
| FXY | −27.10% | −29.60% |

部分 ETF 與論文差距甚大（例如 FXA 論文為 −5.79%，實際為 +7.74%），推測論文採用了不同的計算期間或價格基準。

### 4.4 與論文之比較

與論文一致之處在於模型架構與實驗流程。參數量、資料切分、特徵、標籤、狀態、動作、獎勵結構與 A3C 超參數均依論文設定；PennyLane 與 NumPy 兩種實作的古典 A3C 結果幾乎相同，可相互驗證；古典 A3C 的總報酬與最大回撤也與論文屬同一量級。

與論文不同之處在於結果。QLSTM 預測器的準確率低於論文 6～15 個百分點，但測試集僅 88 筆，此差距部分可能來自抽樣誤差。古典 A3C 的交易次數約為論文的兩倍、勝率高約 26 個百分點，顯示代理人傾向「小賺即出場」，與論文的交易行為不同。此外，NumPy 實作中 QA3C 的報酬低於古典 A3C，與論文的結果相反；PennyLane 實作的 QA3C 結果完成後將再行補充。

交易行為對初始資金、部位大小、獎勵公式、episode 定義等設定相當敏感，而這些正是論文未說明的部分，因此在取得原始設定之前，難以判斷差異來自實作或論文本身。

---

<a id="discussion"></a>
## 5. 討論：資料前視偏誤

重現實驗一時發現，同規模的古典 LSTM 也能達到約 69% 的 DA，且明顯優於隨機漫步。對匯率的隔日方向而言，這個準確率異常地高，因此進一步檢查資料。

檢查結果顯示，Yahoo Finance 匯率日線的收盤價與同一列開盤價的中位數差距僅 1～2 bp，即收盤價約等於該 K 棒開始時的價格；然而最高價與最低價涵蓋整日。因此，以同一列的開高低價預測隔日收盤，等同使用了預測當下尚未發生的資訊。以 EUR/USD 為例，不使用任何模型，僅比較當日高低價中點與收盤價，即可判斷隔日方向 73.1%，與論文報告的 DA 相近。USD/TWD 也有相同現象。

實驗一的訓練樣本數（4,374 筆）與論文完全一致，推測論文同樣使用 Yahoo Finance 的資料。若是如此，論文約 72% 的 DA 可能主要反映資料特性，而非模型的預測能力；QFWP 是否仍優於 QLSTM，需要以修正後的資料重新驗證。

---

<a id="future"></a>
## 6. 待加強與後續工作

1. **以無前視偏誤的資料重跑實驗一。** 將第 i 列的最高價與最低價改為第 i−1 列的值，重新比較 QLSTM、QFWP 與古典 LSTM。若 QFWP 仍顯著優於其他兩者，論文的結論才能在乾淨資料上成立。
2. **確認論文未說明的設定。** 實驗二的交易行為對第 4.1 節所列的設定非常敏感。若能取得原始程式碼或確認這些設定，即可直接比對差異來源。
3. **增加實驗二的統計強度。** QA3C 目前僅有 1 個種子的完整結果，強化學習的變異大，應至少執行 3～5 個種子。預測器的測試集僅 88 筆，可考慮 walk-forward 驗證或延長測試期間。
4. **加入交易成本。** 論文未計交易成本，而本重現的古典 A3C 在測試期間交易約 430 次。應比較 0、0.05%、0.1% 三種成本下的報酬，評估策略的實際可行性。
5. **針對 QLSTM 調整學習率。** 實驗一中 QLSTM 與論文的差距集中在小 batch，應單獨為 QLSTM 掃描學習率，確認差距是否來自訓練設定。

---

<a id="usage"></a>
## 7. 程式使用方式

需要 Python 3.11 或 3.12。A3C 以 fork 建立 worker 行程，請在 macOS、Linux 或 Windows WSL2 執行。

```bash
# 建立環境
python3.12 -m venv .venv-pl && source .venv-pl/bin/activate
pip install -r requirements-pl.txt yfinance

# 下載資料至 data/
python fetch_data_yf.py

# 確認 PennyLane 與 NumPy 實作一致（每行應顯示 OK）
python pl_check.py

# 執行全部實驗（參數為 CPU 核心數）
export OMP_NUM_THREADS=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES   # 僅 macOS 需要
bash run_pl_all.sh 8
```

亦可分別執行各實驗：

```bash
python pl_exp1.py --seeds 10 --jobs 8 --lr 0.05 --opt adam   # 實驗一
python report1.py pl_adam0.05                                # 彙整實驗一

python pl_exp2.py qlstm --seeds 3                            # 實驗二：QLSTM 預測器
python pl_exp2.py rl --agent A3C  --seeds 3                  # 實驗二：古典 A3C
python pl_exp2.py rl --agent QA3C --seeds 1                  # 實驗二：QA3C
python report2.py exp2_pl                                    # 彙整實驗二
```

詳細步驟與參數說明見 [`實驗說明.md`](實驗說明.md) 與 [`PENNYLANE.md`](PENNYLANE.md)。

---

<a id="structure"></a>
## 8. 專案結構

```
├── README.md              本報告
├── REPORT_design.md       論文設定與本重現設定之逐項對照
├── 實驗說明.md             環境架設、資料導入、執行步驟、名詞說明
├── PENNYLANE.md           PennyLane 實作之安裝與執行
├── requirements-pl.txt    所需套件
├── fetch_data_yf.py       下載 Yahoo Finance 資料
│
├── pl_models.py           PennyLane 模型：QLSTM、QFWP、LSTM、QA3C、A3C
├── pl_check.py            PennyLane 與 NumPy 實作之一致性檢查
├── pl_exp1.py             實驗一（PennyLane）
├── pl_exp2.py             實驗二（PennyLane）
├── run_pl_all.sh          執行全部 PennyLane 實驗
│
├── ad.py                  NumPy 自動微分與 state-vector 量子電路模擬
├── models.py              NumPy 模型：QLSTM、QFWP、LSTM
├── common.py              資料讀取、視窗正規化、DA 計算、訓練迴圈
├── exp1a.py  sweep1.py    實驗一（NumPy）、最佳化器與學習率掃描
├── exp2.py                實驗二（NumPy）：資料、交易環境、代理人、測試
├── report1.py report2.py  結果彙整
├── test_ad.py test_models2.py   NumPy 實作之梯度檢查
│
└── results/               每次訓練之結果（JSON）
    ├── exp1a_pl_adam0.05/   實驗一 PennyLane 實作（10 種子 × 6 batch × 3 模型）
    ├── exp1a_paper*/        實驗一 NumPy 實作
    ├── sweep1/              學習率掃描
    ├── exp2_pl/             實驗二 PennyLane 實作
    └── exp2/                實驗二 NumPy 實作
```

---

<a id="references"></a>
## 9. 參考文獻

1. J.-H. Chen, M.-K. Hung, Y.-C. Tsai, S. Y.-C. Chen, "Batched Training for QLSTM vs. QFWP: A System-Oriented Approach to EPC-Aware RMSE-DA," *ICASSP 2026*. arXiv:2512.21820.
2. J.-H. Chen, Y.-C. Huang, Y.-C. Tsai, S. Y.-C. Chen, "Quantum-Enhanced Forecasting for Deep Reinforcement Learning in Algorithmic Trading," *IEEE Globecom Workshops 2025*. arXiv:2509.09176.
3. S. Y.-C. Chen, S. Yoo, Y.-L. L. Fang, "Quantum Long Short-Term Memory," *ICASSP 2022*.
4. S. Y.-C. Chen, "Learning to Program Variational Quantum Circuits with Fast Weights," *IJCNN 2024*.
