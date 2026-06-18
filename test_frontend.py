"""
test_frontend.py
HTML_CONTENT 内の JavaScript ユニットテスト（Node.js 実行）

実行:
    pytest test_frontend.py -v
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ── app.py を重いモジュールなしでインポート ──────────────────────────
def _make_stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


for _n in ["torch", "torchaudio", "soundfile", "pyannote", "pyannote.audio", "huggingface_hub"]:
    sys.modules.setdefault(_n, _make_stub(_n))
_fw = _make_stub("faster_whisper")
_fw.WhisperModel = MagicMock()
sys.modules.setdefault("faster_whisper", _fw)

with patch("faster_whisper.WhisperModel"):
    import app as _app

# ── <script> ブロック抽出 ─────────────────────────────────────────
def _extract_script() -> str:
    m = re.search(r"<script>(.*?)</script>", _app.HTML_CONTENT, re.DOTALL)
    assert m, "<script> block not found"
    return m.group(1)


# Browser API の最小スタブ（Node.js でスクリプトをロードするために必要）
_DOM_STUBS = r"""
const _mkEl = () => ({
    style: {}, textContent: '', innerHTML: '', checked: false, value: '',
    href: '', disabled: false, dataset: {}, files: [],
    addEventListener: () => {},
    classList: { add: () => {}, remove: () => {}, toggle: () => {} },
});
global.document = {
    getElementById: () => _mkEl(),
    querySelectorAll: () => [],
};
global.fetch = () => Promise.resolve({
    json: () => Promise.resolve({ available: false }),
    ok: true,
    text: () => Promise.resolve(''),
});
global.EventSource = class {
    constructor() {}
    close() {}
    set onmessage(v) {}
    set onerror(v) {}
};
global.XMLHttpRequest = class {
    constructor() { this.upload = {}; }
    open() {} send() {}
    set onload(v) {}
    set onerror(v) {}
};
global.FormData = class { append() {} };
global.alert = () => {};
global.setInterval = () => 0;
global.clearInterval = () => {};
global.setTimeout = () => {};
global.window = global;
"""

_SCRIPT_JS = _extract_script()


def run_js(test_code: str):
    """
    DOM スタブ + アプリの JS 全体 + test_code を Node.js で実行し、
    最後の stdout 行を JSON パースして返す。
    test_code 内で console.log(JSON.stringify(result)) すること。
    """
    full = f"{_DOM_STUBS}\n{_SCRIPT_JS}\n{test_code}"
    r = subprocess.run(["node", "-e", full], capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        raise AssertionError(f"Node.js error:\n{r.stderr}")
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    return json.loads(lines[-1]) if lines else None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# escHtml
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEscHtml:
    def test_ampersand(self):
        assert run_js('console.log(JSON.stringify(escHtml("a & b")))') == "a &amp; b"

    def test_less_than(self):
        assert run_js('console.log(JSON.stringify(escHtml("<tag>")))') == "&lt;tag&gt;"

    def test_greater_than(self):
        assert run_js('console.log(JSON.stringify(escHtml("a > b")))') == "a &gt; b"

    def test_multiple_entities(self):
        result = run_js('console.log(JSON.stringify(escHtml("<a href=\\"#\\">link & more</a>")))')
        assert "&lt;" in result
        assert "&amp;" in result
        assert "&gt;" in result

    def test_no_special_chars_unchanged(self):
        assert run_js('console.log(JSON.stringify(escHtml("hello world")))') == "hello world"

    def test_converts_non_string_to_string(self):
        assert run_js('console.log(JSON.stringify(escHtml(42)))') == "42"

    def test_empty_string(self):
        assert run_js('console.log(JSON.stringify(escHtml("")))') == ""

    def test_xss_script_tag_escaped(self):
        result = run_js('console.log(JSON.stringify(escHtml("<script>alert(1)</script>")))')
        assert "<script>" not in result
        assert "&lt;script&gt;" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# markdownToHtml
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestMarkdownToHtml:
    def _run(self, md: str) -> str:
        escaped = md.replace("\\", "\\\\").replace("`", "\\`")
        return run_js(f"console.log(JSON.stringify(markdownToHtml(`{escaped}`)))")

    def test_h1(self):
        assert "<h1>見出し1</h1>" in self._run("# 見出し1")

    def test_h2(self):
        assert "<h2>見出し2</h2>" in self._run("## 見出し2")

    def test_h3(self):
        assert "<h3>見出し3</h3>" in self._run("### 見出し3")

    def test_horizontal_rule(self):
        assert "<hr>" in self._run("---")

    def test_bullet_list_item(self):
        result = self._run("- アイテム")
        assert "<li>アイテム</li>" in result
        assert "<ul>" in result

    def test_table_row(self):
        result = self._run("| A | B |")
        assert "<tr>" in result
        assert "<td>A</td>" in result
        assert "<td>B</td>" in result

    def test_table_separator_row_omitted(self):
        result = self._run("| --- | --- |")
        assert "<tr>" not in result

    def test_table_wrapped_in_table_tag(self):
        result = self._run("| A | B |\n| 1 | 2 |")
        assert "<table>" in result

    def test_bold_text(self):
        assert "<strong>太字</strong>" in self._run("**太字**")

    def test_xss_script_tag_escaped(self):
        result = self._run("<script>alert(1)</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_xss_in_heading_escaped(self):
        result = self._run("# <script>xss</script>")
        assert "<script>" not in result

    def test_empty_string(self):
        assert self._run("") == ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# extractSpeakers (JS版)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestExtractSpeakersJS:
    def test_extracts_basic_speakers(self):
        result = run_js(r'''
            const r = extractSpeakers("[00:00] 話者1: hello\n[00:05] 話者2: world");
            console.log(JSON.stringify(r));
        ''')
        assert "話者1" in result
        assert "話者2" in result

    def test_colon_not_included_in_label(self):
        """Python版と同様、コロンが話者ラベルに含まれないこと"""
        result = run_js(r'''
            const r = extractSpeakers("[00:00] 話者1: hello");
            console.log(JSON.stringify(r));
        ''')
        assert "話者1:" not in result
        assert "話者1" in result

    def test_deduplication(self):
        result = run_js(r'''
            const r = extractSpeakers("[00:00] 話者1: a\n[00:05] 話者1: b");
            console.log(JSON.stringify(r));
        ''')
        assert result.count("話者1") == 1

    def test_preserves_insertion_order(self):
        result = run_js(r'''
            const r = extractSpeakers("[00:00] 話者1: a\n[00:05] 話者2: b\n[00:10] 話者1: c");
            console.log(JSON.stringify(r));
        ''')
        assert result == ["話者1", "話者2"]

    def test_empty_string(self):
        result = run_js('console.log(JSON.stringify(extractSpeakers("")))')
        assert result == []

    def test_no_speakers(self):
        result = run_js('console.log(JSON.stringify(extractSpeakers("[00:00] こんにちは")))')
        assert result == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# getSpeakerColor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGetSpeakerColor:
    def test_first_speaker_gets_s0(self):
        result = run_js('console.log(JSON.stringify(getSpeakerColor("話者1")))')
        assert result == "s0"

    def test_second_speaker_gets_s1(self):
        result = run_js(r'''
            getSpeakerColor("話者1");
            console.log(JSON.stringify(getSpeakerColor("話者2")));
        ''')
        assert result == "s1"

    def test_same_speaker_returns_same_color(self):
        result = run_js(r'''
            const c1 = getSpeakerColor("話者1");
            const c2 = getSpeakerColor("話者1");
            console.log(JSON.stringify(c1 === c2));
        ''')
        assert result is True

    def test_colors_cycle_after_five(self):
        """5色使い切ったら s0 に戻ること"""
        result = run_js(r'''
            getSpeakerColor("A");
            getSpeakerColor("B");
            getSpeakerColor("C");
            getSpeakerColor("D");
            getSpeakerColor("E");
            console.log(JSON.stringify(getSpeakerColor("F")));
        ''')
        assert result == "s0"

    def test_returns_valid_css_class(self):
        result = run_js('console.log(JSON.stringify(getSpeakerColor("test")))')
        assert result in ["s0", "s1", "s2", "s3", "s4"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# renderSpeakerTranscript
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRenderSpeakerTranscript:
    def test_timestamp_rendered(self):
        result = run_js(r'''
            const r = renderSpeakerTranscript("[00:05] 話者1: こんにちは");
            console.log(JSON.stringify(r));
        ''')
        assert "[00:05]" in result

    def test_speaker_tag_has_css_class(self):
        result = run_js(r'''
            const r = renderSpeakerTranscript("[00:00] 話者1: hello");
            console.log(JSON.stringify(r));
        ''')
        assert "speaker-tag" in result
        assert "話者1" in result

    def test_content_rendered(self):
        result = run_js(r'''
            const r = renderSpeakerTranscript("[00:00] 話者1: おはようございます");
            console.log(JSON.stringify(r));
        ''')
        assert "おはようございます" in result

    def test_xss_in_content_escaped(self):
        result = run_js(r'''
            const r = renderSpeakerTranscript("[00:00] 話者1: <script>alert(1)</script>");
            console.log(JSON.stringify(r));
        ''')
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_xss_in_speaker_name_escaped(self):
        result = run_js(r'''
            const r = renderSpeakerTranscript("[00:00] <img src=x>: hello");
            console.log(JSON.stringify(r));
        ''')
        assert "<img" not in result

    def test_unmatched_line_rendered_as_plain(self):
        """タイムスタンプ形式に合わない行はプレーンテキストで表示"""
        result = run_js(r'''
            const r = renderSpeakerTranscript("普通のテキスト");
            console.log(JSON.stringify(r));
        ''')
        assert "普通のテキスト" in result

    def test_multiple_lines_all_rendered(self):
        result = run_js(r'''
            const transcript = "[00:00] 話者1: A\n[00:05] 話者2: B";
            const r = renderSpeakerTranscript(transcript);
            console.log(JSON.stringify(r));
        ''')
        assert "話者1" in result
        assert "話者2" in result
        assert "speaker-line" in result
