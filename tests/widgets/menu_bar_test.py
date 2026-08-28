import inspect
from unittest.mock import Mock

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

from buzz.locale import _
import buzz.widgets.menu_bar as menu_bar_module
from buzz.widgets.menu_bar import MenuBar
from buzz.widgets.preferences_dialog.models.preferences import Preferences
from buzz.widgets.preferences_dialog.preferences_dialog import PreferencesDialog


@pytest.fixture(scope="session")
def qapp_cls():
    return QApplication


class TestMenuBar:
    def test_view_menu_is_between_file_and_help_and_contains_meetings(
        self, qtbot, shortcuts
    ):
        menu_bar = MenuBar(
            shortcuts=shortcuts, preferences=Preferences.load(QSettings())
        )
        qtbot.add_widget(menu_bar)

        menus = menu_bar.actions()
        assert menus[0].text() == _("File")
        assert menus[1].text() == _("View")
        assert menus[2].text().replace("\u200b", "") == _("Help")
        assert menus[1].menu().actions() == [menu_bar.meetings_action]
        assert menu_bar.meetings_action.text() == _("Meetings")

    def test_meetings_action_emits_signal_exactly_once(self, qtbot, shortcuts):
        menu_bar = MenuBar(
            shortcuts=shortcuts, preferences=Preferences.load(QSettings())
        )
        qtbot.add_widget(menu_bar)
        signal_mock = Mock()
        menu_bar.meetings_action_triggered.connect(signal_mock)

        menu_bar.meetings_action.trigger()

        signal_mock.assert_called_once_with()

    def test_menu_bar_does_not_build_meeting_dependencies(self):
        source = inspect.getsource(menu_bar_module)
        assert "QSql" not in source
        assert "MeetingLibraryRepository" not in source
        assert "MeetingLibraryService" not in source

    def test_import_folder_action_emits_signal(self, qtbot, shortcuts):
        menu_bar = MenuBar(
            shortcuts=shortcuts, preferences=Preferences.load(QSettings())
        )
        qtbot.add_widget(menu_bar)

        signal_mock = Mock()
        menu_bar.import_folder_action_triggered.connect(signal_mock)
        menu_bar.import_folder_action.trigger()

        signal_mock.assert_called_once()

    def test_open_preferences_dialog(self, qtbot, shortcuts):
        menu_bar = MenuBar(
            shortcuts=shortcuts, preferences=Preferences.load(QSettings())
        )
        qtbot.add_widget(menu_bar)

        preferences_dialog = menu_bar.findChild(PreferencesDialog)
        assert preferences_dialog is None

        menu_bar.preferences_action.trigger()

        preferences_dialog = menu_bar.findChild(PreferencesDialog)
        assert isinstance(preferences_dialog, PreferencesDialog)
