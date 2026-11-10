import json
import logging

from src.main import JsonFormatter, RequestIdFilter, request_id_context


def create_log_record(message="Test log message"):
    return logging.LogRecord(
        name="src.main",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


def test_json_formatter_outputs_valid_json():
    record = create_log_record()
    record.request_id = "test-request-123"

    formatter = JsonFormatter()
    formatted = formatter.format(record)

    log_data = json.loads(formatted)

    assert log_data["level"] == "INFO"
    assert log_data["logger"] == "src.main"
    assert log_data["message"] == "Test log message"
    assert log_data["request_id"] == "test-request-123"
    assert "timestamp" in log_data


def test_request_id_filter_adds_request_id_from_context():
    token = request_id_context.set("context-request-456")

    try:
        record = create_log_record()

        request_filter = RequestIdFilter()
        result = request_filter.filter(record)

        assert result is True
        assert record.request_id == "context-request-456"

    finally:
        request_id_context.reset(token)