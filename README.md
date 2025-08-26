# generate_images_web

# 🎨 AI 圖像生成器

基於 Stable Diffusion 的多模型圖像生成工具，支援中文輸入自動翻譯。

## ✨ 功能特色

- **三種模型**：MajicMIX Realistic v6、Realistic Vision V5.1、Stable Diffusion v1.5
- **中文支援**：自動翻譯中文提示詞為英文
- **參數調整**：自訂種子、尺寸、步數等
- **批量生成**：一次生成多張圖像
- **Web 介面**：直觀的 Gradio 界面

## 🛠️ 快速開始

### 安裝依賴
```bash
pip install diffusers torch gradio matplotlib openai
```

### 設定 API
在 Google Colab 中設定 Groq API 金鑰，或設定環境變數：
```bash
export GROQ_API_KEY="your-api-key"
```

### 執行程式
```bash
python main.py
```

## 🎯 使用方式

1. **選擇模型**：從下拉選單選擇 AI 模型
2. **輸入提示詞**：描述想要生成的圖像（支援中文）
3. **調整參數**：
   - 圖像尺寸：512/768/1024
   - 生成步數：10-50（建議20-30）
   - 生成張數：1-4張
4. **點擊生成**

## 📝 提示詞範例

**人像**
```
一個微笑的年輕女性，長捲髮，穿著白色襯衫
```

**風景**
```
夕陽下的山脈和湖泊
```

**奇幻**
```
魔法森林中的精靈
```

## ⚙️ 系統需求

- NVIDIA GPU（8GB+ VRAM）
- Python 3.8+
- CUDA 11.0+

## 🔧 常見問題

**記憶體不足**：降低圖像解析度或減少生成張數

**模型載入失敗**：檢查網路連線

**翻譯無效**：確認 Groq API 金鑰設定

## 📚 技術棧

- PyTorch + Diffusers
- Gradio 介面
- Groq API 翻譯

---
