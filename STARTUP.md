# 起動要件

## 動作確認済み環境

| 項目 | バージョン |
|------|-----------|
| OS | Windows 11 Home |
| Python | 3.13.14（Windows Store版） |
| torch | 2.12.0+cpu |
| torchaudio | 2.11.0+cpu |
| faster-whisper | 1.2.1 |
| pyannote.audio | 3.3.2 |
| anthropic SDK | 0.109.2 |
| huggingface_hub | 1.19.0 |
| FastAPI | 0.115.0 |
| uvicorn | 0.30.6 |

---

## 前提ソフトウェア

### FFmpeg
winget でインストール（full_build 版）。full-shared 版は不要。

```powershell
winget install Gyan.FFmpeg
```

インストール後 PowerShell を再起動すること。

---

## パッケージインストール

### 1. requirements.txt（基本パッケージ）

```powershell
python -m pip install -r requirements.txt
```

### 2. 追加パッケージ（requirements.txt に含まれないが必須）

```powershell
python -m pip install matplotlib soundfile torchcodec
```

| パッケージ | 用途 |
|-----------|------|
| matplotlib | pyannote.audio の内部依存 |
| soundfile | torchaudio 2.11 の音声読み込み代替バックエンド |
| torchcodec | pyannote.audio 3.x のインポート時参照（実動作は soundfile が担う） |

> **注意**: torchcodec は Windows の FFmpeg DLL 構成の問題で直接は機能しない。app.py 内のパッチにより soundfile に置き換えられる。

---

## 環境変数（.env）

`.env.example` をコピーして `.env` を作成し、以下を記入する。

```env
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx   # 必須
HF_TOKEN=hf_xxxxxxxx               # 話者識別を使う場合のみ必須
WHISPER_MODEL=large-v3             # tiny / base / small / medium / large-v3
```

---

## 起動コマンド

```powershell
python -m uvicorn app:app --port 8510 --env-file .env
```

ブラウザで `http://localhost:8510` を開く。

初回起動時は Whisper モデルのダウンロード・ロードに **1〜3 分**かかる。
話者識別の初回実行時は pyannote モデルのダウンロードにさらに数分かかる。

---

## 既知の互換性問題と対処

以下の問題はすべて `app.py` 冒頭のモンキーパッチで自動解決済み。手動対応は不要。

| エラー | 原因 | 対処（app.py 内） |
|--------|------|-----------------|
| `torchaudio has no attribute 'AudioMetaData'` | torchaudio 2.11 で削除 | dataclass で復元 |
| `torchaudio has no attribute 'list_audio_backends'` | torchaudio 2.11 で削除 | `lambda: ["soundfile", "sox_io"]` で復元 |
| `torchaudio has no attribute 'info'` | torchaudio 2.11 で削除 | soundfile で代替実装 |
| `TorchCodec is required for load_with_torchcodec` | torchcodec の DLL 読み込み失敗 | `torchaudio.load` を soundfile で置き換え |
| `hf_hub_download() unexpected keyword argument 'use_auth_token'` | huggingface_hub 0.20+ で引数名変更 | `use_auth_token` → `token` に変換するラッパー |
| `Weights only load failed` (torch.load) | PyTorch 2.6+ でデフォルトが `weights_only=True` に変更 | `torch.serialization.load` を `weights_only=False` で上書き |
| `No module named 'matplotlib'` | requirements.txt に未記載 | `pip install matplotlib` |

---

## requirements.txt の推奨更新内容

現在の `requirements.txt` に以下を追記しておくと、次回セットアップ時に漏れがなくなる。

```
matplotlib>=3.0.0
soundfile>=0.12.0
torchcodec>=0.14.0
```
