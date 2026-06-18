# 起動要件

## 動作確認済み環境

| 項目 | バージョン・詳細 |
|------|----------------|
| OS | Windows 11 Home |
| Python | 3.13.14（Windows Store版） |
| GPU | NVIDIA GeForce RTX 4060 |
| NVIDIAドライバ | 595.95（CUDA 13.2対応） |
| CUDA（PyTorch用） | 12.8（cu128） |
| torch | 2.11.0+cu128 |
| torchaudio | 2.11.0+cu128 |
| ctranslate2 | 4.8.0（CUDA対応） |
| faster-whisper | 1.2.1 |
| pyannote.audio | 3.3.2 |
| anthropic SDK | 0.109.2 |
| huggingface_hub | 1.19.0 |
| FastAPI | 0.115.0 |
| uvicorn | 0.30.6 |

---

## 前提条件

### 1. NVIDIAドライバ（GPU使用時）
ドライババージョン **525以上** が必要（CUDA 12.x対応）。  
[nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx) から最新版をインストール。

### 2. FFmpeg
```powershell
winget install Gyan.FFmpeg
```
インストール後 PowerShell を再起動すること。  
**full_build 版を使用。full-shared 版は不要。**

### 3. AIプロバイダーのAPIキー（いずれか1つ必須）

| プロバイダー | 取得先 | 環境変数 |
|------------|--------|---------|
| Anthropic Claude | [console.anthropic.com](https://console.anthropic.com) | `ANTHROPIC_API_KEY` |
| OpenAI | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | `OPENAI_API_KEY` |
| Google Gemini | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) | `GEMINI_API_KEY` |

### 4. HuggingFace Token + モデル利用規約への同意（話者識別を使う場合のみ）
1. [huggingface.co](https://huggingface.co) でアカウント作成
2. Settings → Access Tokens でトークンを発行（`hf_...`）
3. 以下の2モデルの利用規約ページを開き **Agree** をクリック
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

> **この同意をしないと話者識別モデルのダウンロードが403エラーで失敗する。**

---

## セットアップ手順

### Step 1: PyTorch（GPU版）のインストール

requirements.txt より**先に**実行すること（pyannote.audio がtorchに依存するため）。

**GPU環境（CUDA 12.8 / RTX 4060）:**
```powershell
pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

**CPU環境（GPUなし）:**
```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 2: 残りのパッケージをインストール

```powershell
pip install -r requirements.txt
```

### Step 3: 環境変数の設定

`.env.example` をコピーして `.env` を作成し、APIキーを記入する。

```powershell
copy .env.example .env
```

`.env` の内容:
```env
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx   # Anthropic を使う場合
# OPENAI_API_KEY=sk-xxxxxxxx       # OpenAI を使う場合
# GEMINI_API_KEY=AIza-xxxxxxxx     # Google Gemini を使う場合
HF_TOKEN=hf_xxxxxxxx               # 話者識別を使う場合のみ
WHISPER_MODEL=large-v3             # tiny / base / small / medium / large-v3
```

---

## 起動コマンド

```powershell
python -m uvicorn app:app --port 8510 --env-file .env
```

ブラウザで `http://localhost:8510` を開く。

| タイミング | 所要時間 |
|-----------|---------|
| 初回起動（Whisperモデルのダウンロード） | 数分〜十数分（モデルサイズによる） |
| 2回目以降の起動（モデルロード） | 30〜60秒 |
| 話者識別の初回実行（pyannoteモデルのダウンロード） | 数分 |

---

## 既知の互換性問題と対処

以下はすべて `app.py` 冒頭のモンキーパッチで**自動解決済み**。手動対応は不要。

| エラー | 原因 | 対処 |
|--------|------|------|
| `torchaudio has no attribute 'AudioMetaData'` | torchaudio 2.11 で削除 | dataclass で復元 |
| `torchaudio has no attribute 'list_audio_backends'` | torchaudio 2.11 で削除 | lambda で復元 |
| `torchaudio has no attribute 'info'` | torchaudio 2.11 で削除 | soundfile で代替実装 |
| `TorchCodec is required` | Windows で torchcodec DLL が読めない | `torchaudio.load` を soundfile で置き換え |
| `hf_hub_download() unexpected keyword 'use_auth_token'` | huggingface_hub 0.20+ で引数名変更 | `use_auth_token` → `token` に変換 |
| `Weights only load failed` | PyTorch 2.6+ で `weights_only=True` がデフォルト化 | `torch.serialization.load` を上書き |
