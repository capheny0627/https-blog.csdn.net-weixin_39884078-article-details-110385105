# 人工智慧期中報告：tensorflow圖形檢測\_使用Google Colab使用Tensorflow進行自訂物件偵測

## 組員：11228028楊承偉、11228033陳寬恆

本專案為人工智慧課程期中報告，旨在記錄我們如何在 **Google Colab** 免費算力環境下，從 **前期資料蒐集與標註** 開始，一路到使用 **TensorFlow/Keras** 建立、訓練與測試一個自訂的物件偵測模型，用以精準辨識日常生活中的「打火機（Lighter）」。本實作採用遷移學習（Transfer Learning）架構，並透過手動解析 XML 標註檔、清洗異常資料、資料歸一化與模型訓練，完整呈現端到端的機器學習專案開發生命週期。

## 一、 專案開發流程總覽

下表整理了我們團隊從無到有的完整執行步驟與核心技術對照：

|階段|步驟名稱|核心工具 / 套件|主要目的與內容|
|-|-|-|-|
|**階段一**|前期資料蒐集與標註|網路圖片搜尋、LabelImg|透過網路尋找並下載 90\~100 張打火機圖片，並透過 Pascal VOC 格式標註邊界框。|
|**階段二**|資料夾結構與路徑掛載|Google Colab, OS 模組|將資料上傳至 Colab 工作目錄中，建立統一的訓練資料夾架構。|
|**階段三**|資料自動過濾與配對|Python Set 交集|比對 `.jpg` 與 `.xml` 檔名，自動過濾並剔除未配對的異常檔案。|
|**階段四**|XML 標註解析與座標轉換|`xml.etree.ElementTree`|讀取標註檔內的極端座標，並將其轉換為 `0\~1` 的相對比例。|
|**階段五**|影像前處理與正規化|OpenCV, NumPy|將影像縮放至 `224x224` 像素，並進行像素值正規化（`/ 255.0`）。|
|**階段六**|建立遷移學習模型|TensorFlow / Keras|載入 `MobileNetV2` 預訓練骨幹，凍結特徵層並外接 4 節點的迴歸輸出層。|
|**階段七**|模型訓練與成效記錄|Model Fit, Adam, MSE|設定 20 個 Epochs、`batch\_size=8` 進行訓練，監控 Loss 與 MAE 數值。|
|**階段八**|儲存模型與視覺化驗證|Matplotlib, OpenCV|匯出 `.h5` 模型權重，並隨機抽樣進行推論，繪製綠色預測邊界框。|

## 二、 前期準備：資料蒐集與標註流程

1. **影像蒐集（Data Acquisition）**：

   * 透過網路搜尋引擎尋找並下載約 90\~100 張打火機圖片，涵蓋不同角度、光影背景與款式，以豐富模型的訓練多樣性。
2. **標註工具選擇（Annotation Tool）**：

   * 使用開源影像標註工具 **LabelImg** 進行手動標註。
   * <img width="1084" height="448" alt="image" src="https://github.com/user-attachments/assets/3d9fab99-be42-49ff-aee8-bd83966dc156" />

   * <img width="1919" height="1023" alt="image" src="https://github.com/user-attachments/assets/3ec5314c-f95f-4b5b-b04a-c7b4e2548592" />

3. **標註格式與規範**：

   * 標註格式選擇 **Pascal VOC**（會產生與圖片同檔名的 `.xml` 標註檔案）。
   * 框選物件時，必須緊貼打火機的邊緣（包含主體與上方火苗或按壓處），並將類別標籤（Class Name）統一設定為 `lighter`。

## 三、 專案架構與目錄說明

將蒐集並標註完成的檔案上傳至 Colab 工作目錄中，確保資料夾結構如下：

```
my\_object\_detection/
│
├── train/                      # 訓練集圖片與 Pascal VOC 格式 XML 標註檔資料夾
│   ├── images (1).jpg
│   ├── images (1).xml
│   ├── images (2).jpg
│   ├── images (2).xml
│   └── ...
│
├── lighter\_detector\_model.h5      # 訓練完成並匯出的模型權重檔
└── README.md                      # 本專案說明文件

```

## 四、 開發環境與依賴套件

* **開發平台**：Google Colab (建議啟用 GPU 硬體加速：`執行階段` -> `變更執行階段類型` -> `硬體加速器` 選擇 T4 GPU)
* **核心框架**：TensorFlow 2.x / Keras
* **影像處理與視覺化**：OpenCV (`cv2`), NumPy, Matplotlib, Python 內建 XML 解析器 (`xml.etree.ElementTree`)

## 五、 詳細實作步驟與程式碼

### 步驟 1：掛載 Google 雲端硬碟與確認資料夾路徑

首先，將 Google 雲端硬碟掛載到 Colab 中，並透過程式碼檢查 `train` 資料夾路徑是否正確。

```
import os
import xml.etree.ElementTree as ET

# 若使用 Google 雲端硬碟，可執行以下指令掛載（若檔案直接上傳至 Colab 暫存區則可略過）
# from google.colab import drive
# drive.mount('/content/drive')

# 設定並檢查訓練資料夾路徑
train\_dir = 'my\_object\_detection/train'
if os.path.exists(train\_dir):
    print("Train 資料夾內容檢查成功，路徑存在！")
    print("部分檔案預覽:", os.listdir(train\_dir)\[:5])
else:
    print("找不到指定的資料夾，請檢查路徑是否正確。")

```

### 步驟 2：讀取資料並自動過濾未配對的檔案

在實際操作中，常會遇到 `.jpg` 圖片數量與 `.xml` 標註檔數量產生落差。透過 Python 的集合（Set）交集功能，可以自動篩選出「圖文雙全」的有效資料組。

```
import glob

# 取得資料夾內所有的 jpg 與 xml 檔案路徑
train\_images = glob.glob(os.path.join(train\_dir, '\*.jpg'))
train\_xmls = glob.glob(os.path.join(train\_dir, '\*.xml'))

print(f"原始訓練集的圖片數量: {len(train\_images)}")
print(f"原始訓練集的 XML 標註數量: {len(train\_xmls)}")

# 萃取不含副檔名的檔案名稱作為集合
image\_names = set(\[os.path.splitext(os.path.basename(p))\[0] for p in train\_images])
xml\_names = set(\[os.path.splitext(os.path.basename(p))\[0] for p in train\_xmls])

# 取交集：只留下雙方都有對應的檔名
valid\_names = image\_names.intersection(xml\_names)
print(f"經過自動過濾與配對後，有效的訓練資料總數: {len(valid\_names)}")

```

### 步驟 3：解析 XML 標註檔與座標驗證

透過 `xml.etree.ElementTree` 模組，深入解析標註檔內部的物件標籤與邊界框極端座標（`xmin`, `ymin`, `xmax`, `ymax`）。

```
# 測試解析其中一個配對成功的 xml 檔案
sample\_xml = train\_xmls\[0]
tree = ET.parse(sample\_xml)
root = tree.getroot()

print(f"測試檔案名稱: {root.find('filename').text}")
for member in root.findall('object'):
    class\_name = member.find('name').text
    xmin = int(member.find('bndbox/xmin').text)
    ymin = int(member.find('bndbox/ymin').text)
    xmax = int(member.find('bndbox/xmax').text)
    ymax = int(member.find('bndbox/ymax').text)
    print(f"物件標籤: {class\_name} | 原始像素座標 -> xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}")

```

### 步驟 4：影像前處理與座標歸一化 (Normalization)

將每張圖片讀取並縮放至固定尺寸 `224x224` 像素，像素值除以 `255.0` 進行正規化；同時將絕對像素座標除以原圖寬高，轉換為 `0\~1` 的相對比例。

```
import numpy as np
from tensorflow.keras.preprocessing.image import load\_img, img\_to\_array

images\_data = \[]
bboxes\_data = \[]

# 走訪所有 xml 檔案進行全面資料打包
for xml\_path in train\_xmls:
    tree = ET.parse(xml\_path)
    root = tree.getroot()
    
    filename = root.find('filename').text
    img\_path = os.path.join(train\_dir, filename)
    
    if os.path.exists(img\_path):
        # 讀取圖片並縮放至 224x224
        img = load\_img(img\_path, target\_size=(224, 224))
        img\_array = img\_to\_array(img) / 255.0  # 正規化像素
        
        member = root.find('object')
        if member is not None:
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            
            # 取得座標並轉換為 0\~1 的相對比例 \[xmin, ymin, xmax, ymax]
            xmin = int(member.find('bndbox/xmin').text) / width
            ymin = int(member.find('bndbox/ymin').text) / height
            xmax = int(member.find('bndbox/xmax').text) / width
            ymax = int(member.find('bndbox/ymax').text) / height
            
            images\_data.append(img\_array)
            bboxes\_data.append(\[xmin, ymin, xmax, ymax])

# 轉換為 NumPy 陣列
X = np.array(images\_data)
y\_bbox = np.array(bboxes\_data)

print("已完成所有資料轉換！")
print("影像特徵陣列形狀 (X):", X.shape)
print("目標座標陣列形狀 (y\_bbox):", y\_bbox.shape)

```

### 步驟 5：建構基於 MobileNetV2 的遷移學習模型

採用輕量且高效的 `MobileNetV2` 作為特徵萃取骨幹，預先載入 ImageNet 權重，將特徵層凍結後，外接全連接層與 4 節點的迴歸輸出層。

```
import tensorflow as tf
from tensorflow.keras import layers, models

# 載入 MobileNetV2 基礎模型（不包含頂層分類器）
base\_model = tf.keras.applications.MobileNetV2(
    input\_shape=(224, 224, 3), 
    include\_top=False, 
    weights='imagenet'
)
base\_model.trainable = False  # 凍結預訓練權重，加速訓練

# 建立 Functional API 模型架構
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base\_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)

# 輸出層：4個節點的迴歸輸出，對應邊界框座標
bbox\_output = layers.Dense(4, activation='sigmoid', name='bbox\_output')(x)

model = models.Model(inputs=inputs, outputs=bbox\_output)

# 編譯模型：使用 adam 優化器、均方誤差 (MSE) 作為迴歸損失函數
model.compile(optimizer='adam', loss='mse', metrics=\['mae'])

# 檢視模型結構摘要
model.summary()

```

### 步驟 6：執行模型訓練

設定總訓練輪數為 `epochs=20`、批次大小 `batch\_size=8`，並自動切分 20% 的資料作為驗證集。

```
# 開始訓練模型
history = model.fit(
    X, y\_bbox, 
    epochs=20, 
    batch\_size=8, 
    validation\_split=0.2
)

```

**實際訓練過程與效能數據記錄：**

* **Epoch 3/20**: `loss: 0.0227` - `mae: 0.1066` - `val\_loss: 0.0283` - `val\_mae: 0.1237`
* **Epoch 10/20**: `loss: 0.0063` - `mae: 0.0505` - `val\_loss: 0.0145` - `val\_mae: 0.0934`
* **Epoch 20/20**: `loss: 0.0017` - `mae: 0.0300` - `val\_loss: 0.0130` - `val\_mae: 0.0875`

### 步驟 7：模型儲存與視覺化測試驗證

將模型匯出儲存為 `.h5` 檔案，並隨機挑選圖片進行實際推論，透過 OpenCV 畫出綠色邊界框並以 Matplotlib 視覺化展示。

```
import matplotlib.pyplot as plt
import cv2

# 1. 儲存模型至當前工作目錄
model.save('lighter\_detector\_model.h5')
print("模型已成功儲存為 lighter\_detector\_model.h5")

# 2. 隨機挑選一張圖片進行推論測試
test\_img\_path = train\_images\[0]
original\_img = cv2.imread(test\_img\_path)
h, w, \_ = original\_img.shape

# 影像預處理
img = load\_img(test\_img\_path, target\_size=(224, 224))
img\_array = img\_to\_array(img) / 255.0
img\_input = np.expand\_dims(img\_array, axis=0)

# 模型預測邊界框座標 (輸出為 0\~1 的比例)
pred\_box = model.predict(img\_input)\[0]  

# 將比例還原成真實圖片的像素座標
xmin = int(pred\_box\[0] \* w)
ymin = int(pred\_box\[1] \* h)
xmax = int(pred\_box\[2] \* w)
ymax = int(pred\_box\[3] \* h)

# 在原始圖片上繪製綠色預測邊界框 (厚度為 3)
cv2.rectangle(original\_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

# 轉換顏色通道並顯示結果
original\_img\_rgb = cv2.cvtColor(original\_img, cv2.COLOR\_BGR2RGB)
plt.figure(figsize=(6, 6))
plt.imshow(original\_img\_rgb)
plt.axis('off')
plt.show()

```

## 六、 參考資料來源

* [CSDN 技術部落格參考文章](https://blog.csdn.net/weixin_39884078/article/details/110385105)

##\七、 結論與心得

透過這次的人工智慧期中專案，我們第一次完整體驗了從無到有打造一個物件偵測模型的過程。一開始要自己上網找圖片、用 LabelImg 一張張拉框標註，才深刻體會到人工智慧背後「資料為王」的道理——乾淨且正確的資料對於模型訓練真的超級重要！

在程式實作方面，一開始對於資料預處理、座標正規化以及 XML 標註檔的解析覺得有點繁瑣，但跟著步驟一步步做下來後，漸漸理解了深度學習模型吃進去的資料格式與數學原理。特別是使用 Google Colab 的免費 GPU 進行遷移學習（MobileNetV2），不僅訓練速度飛快，而且只用少少的 90 幾張圖片就能讓模型在 20 個回合內順利收斂。

最後看到模型成功在測試圖片上畫出精準的綠色邊界框時，真的一瞬間很有成就感！這次實作不僅讓我們把課堂上學到的 TensorFlow 技巧實際應用出來，也解決了許多實務上會遇到的檔案配對與維度轉換問題，對未來繼續學習電腦視覺（Computer Vision）打下了很好的信心與基礎。

