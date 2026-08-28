from unittest.mock import Mock, patch

from buzz.widgets.application import _build_main_window


def test_build_main_window_composes_both_services_with_same_database() -> None:
    database = object()
    transcription_dao = object()
    transcription_segment_dao = object()
    transcription_service = object()
    meeting_repository = object()
    meeting_service = object()
    main_window = Mock(name="main_window")

    with (
        patch(
            "buzz.widgets.application.TranscriptionDAO",
            return_value=transcription_dao,
        ) as transcription_dao_type,
        patch(
            "buzz.widgets.application.TranscriptionSegmentDAO",
            return_value=transcription_segment_dao,
        ) as transcription_segment_dao_type,
        patch(
            "buzz.widgets.application.TranscriptionService",
            return_value=transcription_service,
        ) as transcription_service_type,
        patch(
            "buzz.widgets.application.QSqlMeetingLibraryRepository",
            return_value=meeting_repository,
        ) as meeting_repository_type,
        patch(
            "buzz.widgets.application.MeetingLibraryService",
            return_value=meeting_service,
        ) as meeting_service_type,
        patch(
            "buzz.widgets.application.MainWindow", return_value=main_window
        ) as main_window_type,
    ):
        result = _build_main_window(database)

    transcription_dao_type.assert_called_once_with(database)
    transcription_segment_dao_type.assert_called_once_with(database)
    transcription_service_type.assert_called_once_with(
        transcription_dao, transcription_segment_dao
    )
    meeting_repository_type.assert_called_once_with(database)
    meeting_service_type.assert_called_once_with(meeting_repository)
    main_window_type.assert_called_once_with(transcription_service, meeting_service)
    assert result is main_window
