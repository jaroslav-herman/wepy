import pandas as pd
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch
from pathlib import Path

import wepy
import wepy.basics as basics
from wepy.iv_curve import IV_curves_data


class FakeMPR:
    def __init__(self, path, error_on_unknown_column=True):
        self.data = pd.DataFrame({"value": range(6)})
        self.loop_index = FakeMPR.loop_index


class TestMPRLoops(unittest.TestCase):
    def test_read_mpr_adds_sanitized_loop_column(self):
        cases = [
            (None, [0, 0, 0, 0, 0, 0]),
            ([0], [0, 0, 0, 0, 0, 0]),
            ([0, 2, 4], [0, 0, 1, 1, 2, 2]),
            ([0, 3, 6], [0, 0, 0, 1, 1, 1]),
            ([-2, 0, 3, 3, 99], [0, 0, 0, 1, 1, 1]),
        ]
        for loop_index, expected in cases:
            with self.subTest(loop_index=loop_index):
                FakeMPR.loop_index = loop_index
                with patch.object(basics, "MPRfile", FakeMPR):
                    data = basics.read_mpr("measurement.mpr")
                self.assertEqual(data["loop"].tolist(), expected)

    def test_read_file_delegates_mpr_to_loop_reader(self):
        expected = pd.DataFrame({"value": [1], "loop": [0]})
        with patch.object(basics, "read_mpr", return_value=expected) as reader:
            result = basics.read_file("measurement.mpr")

        reader.assert_called_once_with(
            "measurement.mpr", error_on_unknown_column=True
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_read_file_safe_skips_empty_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "empty.mpr"
            path.touch()
            with patch.object(basics, "read_file") as reader:
                self.assertIsNone(basics.read_file_safe(path, warn=False))
            reader.assert_not_called()

    def test_public_import_and_iv_curve_extraction(self):
        self.assertTrue(callable(wepy.read_mpr))
        data = pd.DataFrame(
            {
                "cycle number": [1, 1, 1, 2, 2, 2],
                "control/V": [1.5, 1.6, 1.7, 1.5, 1.6, 1.7],
                "<I>/mA": [100, 200, 300, 110, 210, 310],
            }
        )
        voltages, currents = IV_curves_data(data)
        self.assertEqual(len(voltages), 2)
        self.assertEqual(len(currents), 2)
        self.assertEqual(voltages[0].tolist(), [1.5, 1.6])
        self.assertEqual(currents[1].tolist(), [110, 210])


if __name__ == "__main__":
    unittest.main()
