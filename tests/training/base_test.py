import unittest
from src.training.base import History

METRIC_NAME = "My metric name"
METRIC_VALUE = 14
ANOTHER_METRIC_VALUE = 18
FILE_NAME = "/tmp/file_name"


class HistoryTest(unittest.TestCase):
    def test_whenRecordMetric_shouldBeLogged(self):
        history = History()

        history.record(METRIC_NAME, METRIC_VALUE)
        history.record(METRIC_NAME, ANOTHER_METRIC_VALUE)

        self.assertEqual(history.logs[METRIC_NAME][0], METRIC_VALUE)
        self.assertEqual(history.logs[METRIC_NAME][1], ANOTHER_METRIC_VALUE)

    def test_shouldBeAbleToSave(self):
        history = History()
        history.record(METRIC_NAME, METRIC_VALUE)
        history.save(FILE_NAME)

        loaded_history = History.load(FILE_NAME)

        self.assertEqual(loaded_history.logs, history.logs)
