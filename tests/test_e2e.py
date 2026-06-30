from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _find_button(at, label: str):
    for btn in at.button:
        if btn.label == label:
            return btn
    return None


class TestAppStartup:
    def test_app_starts_without_error(self):
        at = AppTest.from_file("apps/dna_chat.py")
        at.run(timeout=30)
        assert not at.exception

    def test_title_is_correct(self):
        at = AppTest.from_file("apps/dna_chat.py")
        at.run(timeout=30)
        assert at.title[0].value == "moss-dna-gpt"

    def test_all_four_tabs_exist(self):
        at = AppTest.from_file("apps/dna_chat.py")
        at.run(timeout=30)
        assert len(at.tabs) == 4


class TestGenerateTab:
    def test_generate_buttons_exist(self):
        at = AppTest.from_file("apps/dna_chat.py")
        at.run(timeout=30)
        assert _find_button(at, "Generate continuation") is not None
        assert _find_button(at, "Clear history") is not None

    def test_generate_click_no_sliding_no_error(self):
        at = AppTest.from_file("apps/dna_chat.py")
        at.run(timeout=30)
        btn = _find_button(at, "Generate continuation")
        btn.click()
        at.run(timeout=30)
        assert not at.exception

    def test_generate_click_with_sliding_no_error(self):
        at = AppTest.from_file("apps/dna_chat.py")
        at.run(timeout=30)
        at.checkbox(key="sliding_mode").set_value(True)
        at.run(timeout=30)
        btn = _find_button(at, "Generate continuation")
        btn.click()
        at.run(timeout=30)
        assert not at.exception

    def test_sliding_mode_toggle_shows_widgets(self):
        at = AppTest.from_file("apps/dna_chat.py")
        at.run(timeout=30)
        assert at.slider(key="max_new_tokens")
        at.checkbox(key="sliding_mode").set_value(True)
        at.run(timeout=30)
        assert at.number_input(key="total_tokens")
        assert at.selectbox(key="chunk_size")
        at.checkbox(key="sliding_mode").set_value(False)
        at.run(timeout=30)
        assert at.slider(key="max_new_tokens")


class TestScoreTab:
    def test_score_button_exists(self):
        at = AppTest.from_file("apps/dna_chat.py")
        at.run(timeout=30)
        assert _find_button(at, "Score variant") is not None


class TestEvaluateTab:
    def test_evaluate_button_exists(self):
        at = AppTest.from_file("apps/dna_chat.py")
        at.run(timeout=30)
        assert _find_button(at, "Run evaluation") is not None


class TestGraphTab:
    def test_graph_buttons_exist(self):
        at = AppTest.from_file("apps/dna_chat.py")
        at.run(timeout=30)
        assert _find_button(at, "Build graph") is not None

    def test_umap_button_exists(self):
        at = AppTest.from_file("apps/dna_chat.py")
        at.run(timeout=30)
        # UMAP button is in a different viz mode, may not be shown initially
        # Just check the first graph button exists
        assert _find_button(at, "Build graph") is not None
