import os

import pytest
import whisper
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QComboBox, QPushButton
from pytestqt.qtbot import QtBot

from buzz.locale import _
from buzz.model_loader import (
    ModelDownloader,
    WhisperModelSize,
    get_expected_whisper_model_size,
    get_whisper_file_path,
)
from buzz.widgets.preferences_dialog.models_preferences_widget import (
    ModelsPreferencesWidget,
)


@pytest.fixture(scope="session")
def qapp_cls():
    return QApplication


class TestModelsPreferencesWidget:
    @pytest.fixture(autouse=True)
    def isolate_model_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("buzz.model_loader.model_root_dir", str(tmp_path))

    @pytest.fixture
    def fake_model_download(self, monkeypatch):
        requests = []

        def download_model(downloader, url, file_path, expected_sha256):
            requests.append((url, file_path, expected_sha256))
            expected_size = get_expected_whisper_model_size(
                downloader.model.whisper_model_size
            )
            assert expected_size is not None
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as model_file:
                model_file.truncate(expected_size)
            return True

        monkeypatch.setattr(ModelDownloader, "download_model", download_model)
        return requests

    def test_should_show_model_list(self, qtbot):
        widget = ModelsPreferencesWidget()
        qtbot.add_widget(widget)

        first_item = widget.model_list_widget.topLevelItem(0)
        assert first_item.text(0) == _("Downloaded")

        second_item = widget.model_list_widget.topLevelItem(1)
        assert second_item.text(0) == _("Available for Download")

    def test_should_change_model_type(self, qtbot):
        widget = ModelsPreferencesWidget()
        qtbot.add_widget(widget)

        combo_box = widget.findChild(QComboBox)
        assert isinstance(combo_box, QComboBox)
        combo_box.setCurrentText("Faster Whisper")

        first_item = widget.model_list_widget.topLevelItem(0)
        assert first_item.text(0) == _("Downloaded")

        second_item = widget.model_list_widget.topLevelItem(1)
        assert second_item.text(0) == _("Available for Download")

    def test_should_download_model(
        self, qtbot: QtBot, fake_model_download
    ):
        # make progress dialog non-modal to unblock qtbot.wait_until
        widget = ModelsPreferencesWidget(
            progress_dialog_modality=Qt.WindowModality.NonModal
        )
        qtbot.add_widget(widget)

        assert widget.model.get_local_model_path() is None

        available_item = widget.model_list_widget.topLevelItem(1)
        assert available_item.text(0) == _("Available for Download")

        tiny_item = available_item.child(0)
        assert tiny_item.text(0) == "Tiny"
        tiny_item.setSelected(True)

        download_button = widget.findChild(QPushButton, "DownloadButton")
        assert isinstance(download_button, QPushButton)

        assert download_button.text() == _("Download")
        download_button.click()

        def downloaded_model():
            assert not download_button.isVisible()

            _downloaded_item = widget.model_list_widget.topLevelItem(0)
            assert _downloaded_item.childCount() > 0
            assert _downloaded_item.child(0).text(0) == "Tiny"

            _available_item = widget.model_list_widget.topLevelItem(1)
            assert (
                _available_item.childCount() == 0
                or _available_item.child(0).text(0) != "Tiny"
            )

            assert os.path.isfile(widget.model.get_local_model_path())

        qtbot.wait_until(callback=downloaded_model, timeout=60_000)

        assert len(fake_model_download) == 1
        assert fake_model_download[0][0] == whisper._MODELS[WhisperModelSize.TINY.value]
        assert os.path.isfile(fake_model_download[0][1])

    @pytest.fixture
    def default_model_path(self, isolate_model_cache) -> str:
        model_path = get_whisper_file_path(WhisperModelSize.TINY)
        expected_size = get_expected_whisper_model_size(WhisperModelSize.TINY)
        assert expected_size is not None
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as model_file:
            model_file.truncate(expected_size)
        return model_path

    def test_should_show_downloaded_model(self, qtbot, default_model_path):
        widget = ModelsPreferencesWidget()
        widget.show()
        qtbot.add_widget(widget)

        available_item = widget.model_list_widget.topLevelItem(0)
        assert available_item.text(0) == _("Downloaded")

        tiny_item = available_item.child(0)
        assert tiny_item.text(0) == "Tiny"
        tiny_item.setSelected(True)

        delete_button = widget.findChild(QPushButton, "DeleteButton")
        assert delete_button.isVisible()

        show_file_location_button = widget.findChild(
            QPushButton, "ShowFileLocationButton"
        )
        assert show_file_location_button.isVisible()
