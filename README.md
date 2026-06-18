# 議事録自動生成アプリ

音声ファイルをアップロードするだけで、AI が自動で文字起こし・話者識別・議事録生成を行うWebアプリです。

---

## 機能

- **音声文字起こし** — faster-whisper（ローカル実行・無料・プライバシー保護）
- **話者識別** — pyannote.audio による話者分離（誰が話したかを自動判定）
- **AI自動命名** — 会話の文脈（自己紹介・呼びかけ・役職など）から話者名を自動推定
- **話者名登録** — 識別後に各話者へ任意の名前を付与（AI推定結果の手動修正も可能）
- **話者別サマリー** — 各話者の主張・キーワード・スタンスをAIが要約
- **議事録生成** — AI による構造化議事録（Anthropic / OpenAI / Gemini 対応）
- **テンプレート対応** — 独自フォーマットをアップロードしてそのまま出力
- **出力形式** — Markdown / Word（.docx）ダウンロード
- **長時間音声対応** — 自動チャンク分割処理

---

## 技術スタック

| カテゴリ | 技術 |
|---------|------|
| Web フレームワーク | FastAPI + Uvicorn |
| 文字起こし | faster-whisper（OpenAI Whisper ローカル版） |
| 話者識別 | pyannote.audio 3.3.2 |
| 機械学習 | PyTorch |
| 議事録生成 | Anthropic Claude / OpenAI / Google Gemini（選択式） |
| Word 出力 | python-docx |
| 音声変換 | FFmpeg |
| フロントエンド | HTML / CSS / Vanilla JS（組み込み） |

---

## 動作環境

- Windows 10/11
- Python 3.11（Anaconda 推奨）
- FFmpeg（winget でインストール）
- インターネット接続（AI API / HuggingFace モデル取得に必要）

---

## セットアップ

### 1. AIプロバイダーのAPIキーを取得

以下のいずれか1つを取得してください。複数持っている場合は `.env` の `AI_PROVIDER` で切り替えられます。

| プロバイダー | 取得先 | 環境変数 |
|------------|--------|---------|
| Anthropic Claude | [console.anthropic.com](https://console.anthropic.com) | `ANTHROPIC_API_KEY` |
| OpenAI | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | `OPENAI_API_KEY` |
| Google Gemini | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) | `GEMINI_API_KEY` |

**HuggingFace Token（話者識別を使う場合のみ）**
1. [huggingface.co](https://huggingface.co) でアカウント作成
2. Settings → Access Tokens でトークンを発行（`hf_...`）
3. [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) の利用規約に同意

### 2. FFmpeg をインストール

```powershell
winget install Gyan.FFmpeg
```

インストール後、PowerShell を再起動してください。

### 3. Python 環境を構築

```powershell
# conda 環境を作成（Python 3.11 推奨）
conda create -n meeting_transcriber python=3.11 -y
conda activate meeting_transcriber

# 依存パッケージをインストール
pip install -r requirements.txt
```

### 4. 環境変数を設定

`.env.example` をコピーして `.env` を作成し、使用するプロバイダーのAPIキーを記入します。

```powershell
copy .env.example .env
```

**Anthropic を使う場合の `.env` 例：**

```env
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
WHISPER_MODEL=medium
HF_TOKEN=hf_xxxxxxxx
```

**OpenAI を使う場合の `.env` 例：**

```env
OPENAI_API_KEY=sk-xxxxxxxx
OPENAI_MODEL=gpt-4o
WHISPER_MODEL=medium
```

**複数プロバイダーがある場合：**

```env
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxx
AI_PROVIDER=openai   # 使いたいプロバイダーを明示
```

**WHISPER_MODEL の選択肢：**

| モデル | 精度 | 速度 | VRAM |
|--------|------|------|------|
| tiny | 低 | 最速 | 1GB |
| base | やや低 | 速い | 1GB |
| small | 普通 | 普通 | 2GB |
| medium | 高（推奨） | 遅い | 5GB |
| large-v3 | 最高 | 最遅 | 10GB |

### 5. 起動

```powershell
$condaPython = "$env:USERPROFILE\anaconda3\envs\meeting_transcriber\python.exe"
& $condaPython -m uvicorn app:app --port 8510 --env-file .env
```

ブラウザで [http://localhost:8510](http://localhost:8510) を開きます。

---

## 使い方

### 基本（議事録生成）

1. ブラウザで `http://localhost:8510` を開く
2. 音声ファイルをドラッグ＆ドロップ（または「クリックして選択」）
3. 「🚀 議事録を生成」をクリック
4. 完了後、**議事録タブ** と **文字起こしタブ** で結果を確認
5. Markdown または Word ファイルをダウンロード

### 話者識別を使う

1. 「👥 話者識別を有効にする」トグルをON
2. 音声ファイルをアップロードして生成
3. 完了後、話者名入力パネルが表示される
4. 「🤖 AIが自動で名前を推定」をクリックすると、会話の文脈から話者名を推定して入力欄にセット
5. 必要に応じて手動で名前を修正・入力（例：話者1 → 田中）
6. 「✨ 名前を適用して話者別サマリーを生成」をクリック
7. **話者別サマリータブ** で各話者の発言要約を確認

### 議事録テンプレートを使う

1. 「📋 議事録テンプレート（任意）」の「ファイルを選択」をクリック
2. Markdown または テキストファイルを選択（`sample_template.md` を参考に作成）
3. テンプレートに除外ルール・フォーマット指示を記述することで細かい制御が可能

**テンプレートにルールを書く例：**

```markdown
## 出力ルール
- プレゼン発表者の発言は除外する
- 質疑応答のみ記録する
- 敬語に統一する
```

---

## 対応音声フォーマット

MP3 / MP4 / M4A / WAV / OGG / FLAC / WebM / AAC

---

## API 料金の目安

議事録生成は選択したプロバイダーの従量課金が発生します。文字起こし（Whisper）はローカル実行のため無料です。

| プロバイダー | モデル | 1時間音声の目安 |
|------------|--------|--------------|
| Anthropic | claude-opus-4-8 | 約 $0.10〜0.30（約15〜45円） |
| OpenAI | gpt-4o | 約 $0.05〜0.20（約8〜30円） |
| Google | gemini-1.5-pro | 約 $0.03〜0.15（約5〜23円） |

---

## テスト

127件のテストが用意されています。

```powershell
# 全テストを実行
pytest test_app.py test_frontend.py -v
```

| テストファイル | 件数 | 内容 |
|--------------|------|------|
| `test_app.py` | 89件 | Python バックエンド（全エンドポイント・ユーティリティ関数） |
| `test_frontend.py` | 38件 | HTML 内 JavaScript 関数（Node.js 実行） |

> `test_frontend.py` の実行には Node.js が必要です。

---

## ファイル構成

```
meeting_transcriber/
├── app.py               # メインアプリケーション
├── requirements.txt     # Python 依存パッケージ
├── test_app.py          # バックエンドテスト（89件）
├── test_frontend.py     # フロントエンド JS テスト（38件）
├── .env                 # APIキー設定（Git管理外）
├── .env.example         # 環境変数テンプレート
├── .gitignore
├── sample_template.md              # 汎用議事録テンプレート
├── template_class_presentation.md  # 授業内発表用テンプレート
└── outputs/                        # 生成ファイルの保存先（Git管理外）
```
