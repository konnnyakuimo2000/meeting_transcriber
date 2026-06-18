"""
test_app.py
app.py のユニットテスト・インテグレーションテスト

実行:
    pip install pytest pytest-asyncio httpx
    pytest test_app.py -v
"""
from __future__ import annotations

import json
import sys
import types
import unittest.mock as mock
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ── 重いモジュールをインポート前にスタブ化 ────────────────────────────
# WhisperModel / pyannote / torch など GPU依存ライブラリを差し替える
def _make_stub_module(name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# faster_whisper
_fw = _make_stub_module("faster_whisper")
_fw.WhisperModel = MagicMock()
sys.modules.setdefault("faster_whisper", _fw)

# torch / torchaudio / soundfile（pyannoteが要求する）
for _name in ["torch", "torchaudio", "soundfile", "pyannote", "pyannote.audio",
              "huggingface_hub"]:
    sys.modules.setdefault(_name, _make_stub_module(_name))

# アプリ本体をインポート（上記スタブが差し込まれた状態で）
with patch("faster_whisper.WhisperModel"):
    import app as _app  # noqa: E402


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. _assign_speaker_labels
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestAssignSpeakerLabels:
    def test_basic_assignment(self):
        whisper = [{"start": 0.0, "end": 2.0, "text": "hello"}]
        diarize = [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"}]
        result = _app._assign_speaker_labels(whisper, diarize)
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_selects_best_overlap(self):
        """2つの話者セグメントのうち重なりが大きい方を選ぶ"""
        whisper = [{"start": 1.0, "end": 3.0, "text": "hi"}]
        diarize = [
            {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"},  # overlap=0.5
            {"start": 1.5, "end": 4.0, "speaker": "SPEAKER_01"},  # overlap=1.5
        ]
        result = _app._assign_speaker_labels(whisper, diarize)
        assert result[0]["speaker"] == "SPEAKER_01"

    def test_empty_diarize_returns_fallback(self):
        """話者セグメントが空のとき話者?になる"""
        whisper = [{"start": 0.0, "end": 1.0, "text": "test"}]
        result = _app._assign_speaker_labels(whisper, [])
        assert result[0]["speaker"] == "話者?"

    def test_empty_whisper_returns_empty(self):
        diarize = [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}]
        result = _app._assign_speaker_labels([], diarize)
        assert result == []

    def test_preserves_original_fields(self):
        whisper = [{"start": 0.5, "end": 1.5, "text": "abc"}]
        diarize = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]
        result = _app._assign_speaker_labels(whisper, diarize)
        assert result[0]["text"] == "abc"
        assert result[0]["start"] == 0.5
        assert result[0]["end"] == 1.5

    def test_multiple_segments(self):
        whisper = [
            {"start": 0.0, "end": 2.0, "text": "one"},
            {"start": 5.0, "end": 7.0, "text": "two"},
        ]
        diarize = [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
            {"start": 4.0, "end": 8.0, "speaker": "SPEAKER_01"},
        ]
        result = _app._assign_speaker_labels(whisper, diarize)
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_01"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. _format_speaker_transcript
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFormatSpeakerTranscript:
    def test_maps_speaker_to_numbered_label(self):
        segs = [{"start": 0.0, "end": 2.0, "text": "hello", "speaker": "SPEAKER_00"}]
        result = _app._format_speaker_transcript(segs)
        assert "話者1" in result
        assert "hello" in result

    def test_same_speaker_keeps_same_number(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": "a", "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "text": "b", "speaker": "SPEAKER_00"},
        ]
        result = _app._format_speaker_transcript(segs)
        assert result.count("話者1") == 2
        assert "話者2" not in result

    def test_different_speakers_get_incrementing_numbers(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": "a", "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "text": "b", "speaker": "SPEAKER_01"},
        ]
        result = _app._format_speaker_transcript(segs)
        assert "話者1" in result
        assert "話者2" in result

    def test_timestamp_format(self):
        segs = [{"start": 65.0, "end": 70.0, "text": "x", "speaker": "SPEAKER_00"}]
        result = _app._format_speaker_transcript(segs)
        assert "[01:05]" in result

    def test_empty_returns_empty_string(self):
        assert _app._format_speaker_transcript([]) == ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. extract_speakers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestExtractSpeakers:
    def test_basic_extraction(self):
        transcript = "[00:00] 話者1: こんにちは\n[00:05] 話者2: よろしく"
        result = _app.extract_speakers(transcript)
        assert "話者1" in result
        assert "話者2" in result

    def test_colon_not_included_in_label(self):
        """コロンが話者ラベルに含まれないことを確認"""
        transcript = "[00:00] 話者1: hello"
        result = _app.extract_speakers(transcript)
        assert "話者1:" not in result
        assert "話者1" in result

    def test_deduplication(self):
        transcript = "[00:00] 話者1: a\n[00:05] 話者1: b"
        result = _app.extract_speakers(transcript)
        assert result.count("話者1") == 1

    def test_preserves_insertion_order(self):
        transcript = "[00:00] 話者1: a\n[00:05] 話者2: b\n[00:10] 話者1: c"
        result = _app.extract_speakers(transcript)
        assert result == ["話者1", "話者2"]

    def test_empty_transcript(self):
        assert _app.extract_speakers("") == []

    def test_no_speakers(self):
        assert _app.extract_speakers("[00:00] こんにちは") == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. apply_speaker_names
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestApplySpeakerNames:
    def test_basic_replacement(self):
        result = _app.apply_speaker_names("話者1: hello", {"話者1": "田中"})
        assert "田中: hello" == result

    def test_empty_name_skipped(self):
        """値が空文字のエントリは置換しない"""
        result = _app.apply_speaker_names("話者1: hello", {"話者1": ""})
        assert "話者1: hello" == result

    def test_longer_label_replaced_first(self):
        """話者1と話者10が混在するとき、話者10が先に処理されてサブストリング衝突しない"""
        text = "話者1: a\n話者10: b"
        result = _app.apply_speaker_names(text, {"話者1": "田中", "話者10": "鈴木"})
        assert "田中: a" in result
        assert "鈴木: b" in result
        # 「田中0: b」のような誤置換がないこと
        assert "田中0" not in result

    def test_whitespace_stripped_from_name(self):
        result = _app.apply_speaker_names("話者1: hi", {"話者1": "  田中  "})
        assert "田中: hi" == result

    def test_no_map_returns_unchanged(self):
        text = "話者1: hello"
        assert _app.apply_speaker_names(text, {}) == text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. _split_transcript
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSplitTranscript:
    def test_short_text_returns_single_chunk(self):
        text = "line1\nline2\nline3"
        result = _app._split_transcript(text, 1000)
        assert result == [text]

    def test_splits_at_max_chars(self):
        lines = ["x" * 50] * 10  # 合計500文字
        text = "\n".join(lines)
        result = _app._split_transcript(text, 150)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 300  # 多少超えてもよいがチャンクが分割されている

    def test_empty_text_returns_empty_list(self):
        assert _app._split_transcript("", 100) == []

    def test_single_long_line_still_added(self):
        """1行がmax_charsを超えていてもcurrentが空なら追加される"""
        long_line = "a" * 200
        result = _app._split_transcript(long_line, 100)
        assert len(result) == 1
        assert result[0] == long_line

    def test_all_chunks_non_empty(self):
        text = "\n".join([f"line{i}" * 10 for i in range(50)])
        for chunk in _app._split_transcript(text, 100):
            assert chunk.strip() != ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. _extract_text
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestExtractText:
    def _make_block(self, type_: str, text: str = ""):
        b = MagicMock()
        b.type = type_
        b.text = text
        return b

    def test_extracts_text_blocks(self):
        resp = MagicMock()
        resp.content = [self._make_block("text", "hello "), self._make_block("text", "world")]
        assert _app._extract_text(resp) == "hello world"

    def test_ignores_non_text_blocks(self):
        resp = MagicMock()
        resp.content = [
            self._make_block("thinking", "internal"),
            self._make_block("text", "visible"),
        ]
        assert _app._extract_text(resp) == "visible"

    def test_empty_content_returns_empty_string(self):
        resp = MagicMock()
        resp.content = []
        assert _app._extract_text(resp) == ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. auto_identify_speakers（Claude API をモック）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestAutoIdentifySpeakers:
    def _mock_response(self, text: str):
        block = MagicMock()
        block.type = "text"
        block.text = text
        resp = MagicMock()
        resp.content = [block]
        return resp

    def test_returns_speaker_map(self):
        with patch("app.ANTHROPIC_API_KEY", "dummy"), \
             patch("app.anthropic.Anthropic") as mock_client:
            mock_client.return_value.messages.create.return_value = self._mock_response(
                '{"話者1": "田中", "話者2": "佐藤"}'
            )
            result = _app.auto_identify_speakers("話者1: 田中です\n話者2: 佐藤です")
            assert result == {"話者1": "田中", "話者2": "佐藤"}

    def test_returns_empty_when_no_api_key(self):
        with patch("app.ANTHROPIC_API_KEY", ""):
            assert _app.auto_identify_speakers("some text") == {}

    def test_returns_empty_on_malformed_json(self):
        with patch("app.ANTHROPIC_API_KEY", "dummy"), \
             patch("app.anthropic.Anthropic") as mock_client:
            mock_client.return_value.messages.create.return_value = self._mock_response(
                "JSONではないテキスト"
            )
            result = _app.auto_identify_speakers("話者1: hello")
            assert result == {}

    def test_extracts_json_from_mixed_response(self):
        """JSON前後に余分なテキストがあっても抽出できる"""
        with patch("app.ANTHROPIC_API_KEY", "dummy"), \
             patch("app.anthropic.Anthropic") as mock_client:
            mock_client.return_value.messages.create.return_value = self._mock_response(
                'こちらが結果です:\n{"話者1": "田中"}\n以上です。'
            )
            result = _app.auto_identify_speakers("話者1: hello")
            assert result == {"話者1": "田中"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. transcribe_only
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestTranscribeOnly:
    def test_formats_timestamps_correctly(self):
        segments = [
            {"start": 0.0, "end": 2.0, "text": "hello"},
            {"start": 65.0, "end": 67.0, "text": "world"},
        ]
        with patch("app.transcribe_with_segments", return_value=(segments, "ja")):
            text, lang = _app.transcribe_only(Path("dummy.wav"))
        assert "[00:00] hello" in text
        assert "[01:05] world" in text
        assert lang == "ja"

    def test_empty_segments_returns_empty_string(self):
        with patch("app.transcribe_with_segments", return_value=([], "ja")):
            text, lang = _app.transcribe_only(Path("dummy.wav"))
        assert text == ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. APIエンドポイント（FastAPI TestClient）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

try:
    from fastapi.testclient import TestClient
    import io

    _client = TestClient(_app.app, raise_server_exceptions=False)

    class TestDiarizationAvailable:
        def test_returns_false_when_no_hf_token(self):
            with patch("app.HF_TOKEN", ""):
                resp = _client.get("/diarization-available")
            assert resp.status_code == 200
            assert resp.json()["available"] is False

        def test_returns_true_when_hf_token_set(self):
            with patch("app.HF_TOKEN", "dummy_token"):
                resp = _client.get("/diarization-available")
            assert resp.status_code == 200
            assert resp.json()["available"] is True

    class TestDownloadEndpoints:
        def test_download_md_not_found(self):
            resp = _client.get("/download/md/nonexistent_file")
            assert resp.status_code == 404

        def test_download_docx_not_found(self):
            resp = _client.get("/download/docx/nonexistent_file")
            assert resp.status_code == 404

    class TestIdentifySpeakers:
        def test_returns_400_when_no_api_key(self):
            with patch("app.ANTHROPIC_API_KEY", ""):
                resp = _client.post(
                    "/identify-speakers",
                    json={"transcript": "話者1: hello"},
                )
            assert resp.status_code == 400

    class TestTranscribeStart:
        def test_rejects_unsupported_extension(self):
            resp = _client.post(
                "/transcribe-start",
                files={"file": ("test.txt", b"content", "text/plain")},
                data={"use_diarization": "false"},
            )
            assert resp.status_code == 400

        def test_accepts_mp3_and_returns_job_id(self):
            with patch("app._run_job"), \
                 patch("app.asyncio.create_task"):
                resp = _client.post(
                    "/transcribe-start",
                    files={"file": ("test.mp3", b"fake_audio", "audio/mpeg")},
                    data={"use_diarization": "false"},
                )
            assert resp.status_code == 200
            assert "job_id" in resp.json()

    class TestProgressEndpoint:
        def test_unknown_job_id_returns_error(self):
            # SSEストリームの最初のメッセージにエラーが含まれることを確認
            with _client.stream("GET", "/progress/nonexistent-job-id") as resp:
                assert resp.status_code == 200
                first_line = next(
                    (line for line in resp.iter_lines() if line.startswith("data:")),
                    None,
                )
            assert first_line is not None
            data = json.loads(first_line.removeprefix("data: "))
            assert data.get("done") is True
            assert "error" in data

    class TestIndexPage:
        def test_returns_html(self):
            resp = _client.get("/")
            assert resp.status_code == 200
            assert "text/html" in resp.headers["content-type"]
            assert "議事録自動生成" in resp.text

except ImportError:
    pass  # httpx / fastapi.testclient が未インストールの場合はスキップ


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. create_docx
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCreateDocx:
    """create_docx は outputs/ にファイルを書く。各テスト後に削除する。"""

    def _run(self, md: str, title: str = "test_output") -> Path:
        path = _app.create_docx(md, title)
        return path

    def _cleanup(self, path: Path):
        if path.exists():
            path.unlink()

    def test_creates_file(self):
        path = self._run("# タイトル", "test_creates_file")
        try:
            assert path.exists()
            assert path.suffix == ".docx"
        finally:
            self._cleanup(path)

    def test_headings_are_written(self):
        from docx import Document as _Document
        md = "# 見出し1\n## 見出し2\n### 見出し3"
        path = self._run(md, "test_headings")
        try:
            doc = _Document(str(path))
            styles = [p.style.name for p in doc.paragraphs]
            assert any("Heading 1" in s for s in styles)
            assert any("Heading 2" in s for s in styles)
            assert any("Heading 3" in s for s in styles)
        finally:
            self._cleanup(path)

    def test_bullet_list_written(self):
        from docx import Document as _Document
        md = "- アイテム1\n- アイテム2"
        path = self._run(md, "test_bullets")
        try:
            doc = _Document(str(path))
            texts = [p.text for p in doc.paragraphs]
            assert "アイテム1" in texts
            assert "アイテム2" in texts
        finally:
            self._cleanup(path)

    def test_table_written(self):
        from docx import Document as _Document
        md = "| 担当者 | タスク | 期限 |\n|--------|--------|------|\n| 田中 | 資料作成 | 来週 |"
        path = self._run(md, "test_table")
        try:
            doc = _Document(str(path))
            assert len(doc.tables) == 1
            row = doc.tables[0].rows[0]
            assert row.cells[0].text == "担当者"
            assert row.cells[1].text == "タスク"
        finally:
            self._cleanup(path)

    def test_separator_row_skipped(self):
        """Markdownのテーブル区切り行（---|---）はdocxに追加されない"""
        from docx import Document as _Document
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        path = self._run(md, "test_separator")
        try:
            doc = _Document(str(path))
            assert len(doc.tables) == 1
            assert len(doc.tables[0].rows) == 2  # ヘッダー行 + データ行のみ
        finally:
            self._cleanup(path)

    def test_horizontal_rule_written(self):
        from docx import Document as _Document
        md = "---"
        path = self._run(md, "test_hr")
        try:
            doc = _Document(str(path))
            texts = [p.text for p in doc.paragraphs]
            assert any("─" in t for t in texts)
        finally:
            self._cleanup(path)

    def test_plain_paragraph_written(self):
        from docx import Document as _Document
        md = "これは通常の段落です。"
        path = self._run(md, "test_plain")
        try:
            doc = _Document(str(path))
            texts = [p.text for p in doc.paragraphs]
            assert "これは通常の段落です。" in texts
        finally:
            self._cleanup(path)

    def test_empty_lines_skipped(self):
        """空行はdocxに追加されない"""
        from docx import Document as _Document
        md = "段落1\n\n\n段落2"
        path = self._run(md, "test_empty_lines")
        try:
            doc = _Document(str(path))
            texts = [p.text for p in doc.paragraphs if p.text.strip()]
            assert texts == ["段落1", "段落2"]
        finally:
            self._cleanup(path)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. generate_minutes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGenerateMinutes:
    def _mock_text_response(self, text: str):
        block = MagicMock()
        block.type = "text"
        block.text = text
        resp = MagicMock()
        resp.content = [block]
        return resp

    def _mock_stream(self, texts: list[str]):
        stream = MagicMock()
        stream.__enter__ = MagicMock(return_value=stream)
        stream.__exit__ = MagicMock(return_value=False)
        stream.text_stream = iter(texts)
        return stream

    def test_raises_without_api_key(self):
        with patch("app.ANTHROPIC_API_KEY", ""):
            with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
                _app.generate_minutes("テスト")

    def test_short_transcript_no_callback(self):
        """短いテキスト・progress_cbなし → messages.create を1回呼ぶ"""
        with patch("app.ANTHROPIC_API_KEY", "dummy"), \
             patch("app.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = self._mock_text_response("議事録テスト")
            result = _app.generate_minutes("短い文字起こし")
        assert result == "議事録テスト"
        mock_cls.return_value.messages.create.assert_called_once()

    def test_short_transcript_with_callback(self):
        """短いテキスト・progress_cbあり → messages.stream を使う"""
        with patch("app.ANTHROPIC_API_KEY", "dummy"), \
             patch("app.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.stream.return_value = self._mock_stream(["議事", "録"])
            calls = []
            _app.generate_minutes("短い文字起こし", progress_cb=lambda p, l: calls.append(p))
        mock_cls.return_value.messages.stream.assert_called_once()
        assert len(calls) >= 1
        assert calls[0] == 75  # 最初のprogress_cb呼び出し

    def test_template_included_in_system_prompt(self):
        """templateが渡された場合、system_promptにテンプレートが含まれる"""
        with patch("app.ANTHROPIC_API_KEY", "dummy"), \
             patch("app.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = self._mock_text_response("result")
            _app.generate_minutes("文字起こし", template="## カスタムフォーマット")
        call_kwargs = mock_cls.return_value.messages.create.call_args[1]
        assert "カスタムフォーマット" in call_kwargs["system"]

    def test_long_transcript_splits_into_chunks(self):
        """80,000文字超のテキストはチャンク分割して複数回APIを呼ぶ"""
        long_transcript = "a\n" * 50_000  # 約100,000文字
        with patch("app.ANTHROPIC_API_KEY", "dummy"), \
             patch("app.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = self._mock_text_response("chunk summary")
            _app.generate_minutes(long_transcript)
        # チャンク数+最終生成で複数回呼ばれることを確認
        assert mock_cls.return_value.messages.create.call_count >= 2

    def test_long_transcript_progress_callback_called(self):
        """長いテキスト・progress_cbあり → チャンクごとにcbが呼ばれる"""
        long_transcript = "a\n" * 50_000
        with patch("app.ANTHROPIC_API_KEY", "dummy"), \
             patch("app.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = self._mock_text_response("summary")
            mock_cls.return_value.messages.stream.return_value = self._mock_stream(["最終議事録"])
            calls = []
            _app.generate_minutes(long_transcript, progress_cb=lambda p, l: calls.append((p, l)))
        assert any("要約" in label for _, label in calls)
        assert any("最終" in label for _, label in calls)
