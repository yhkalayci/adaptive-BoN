import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from plot_response_char_distributions import (
    load_alignment_means,
    load_coding_means,
    summary_row,
)


class ResponseCharacterDistributionTest(unittest.TestCase):
    def test_alignment_means_use_all_responses_for_each_problem(self):
        records = [
            {
                "JSON_idx": 10,
                "generations": [{"text": "a"}, {"text": "abc"}],
            },
            {
                "JSON_idx": 11,
                "generations": [{"text": "zz"}, {"text": "zzzz"}],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl.gz"
            with gzip.open(path, "wt") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
            ids, means, counts = load_alignment_means(path)

        self.assertEqual(ids, ["10", "11"])
        np.testing.assert_allclose(means, [2.0, 3.0])
        np.testing.assert_array_equal(counts, [2, 2])

    def test_coding_means_use_every_cached_character_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coding.npz"
            np.savez(
                path,
                ids=np.asarray(["p0", "p1"]),
                chars=np.asarray([[10, 20, 30], [2, 4, 6]]),
            )
            ids, means, counts = load_coding_means(path)

        self.assertEqual(ids, ["p0", "p1"])
        np.testing.assert_allclose(means, [20.0, 4.0])
        np.testing.assert_array_equal(counts, [3, 3])

    def test_summary_reports_distribution_over_problem_means(self):
        values = np.asarray([1.0, 3.0, 5.0])
        counts = np.asarray([4, 4, 4])
        row = summary_row("test", "Test", values, counts)

        self.assertEqual(row["problems"], 3)
        self.assertEqual(row["responses_per_problem_min"], 4)
        self.assertEqual(row["responses_per_problem_max"], 4)
        self.assertEqual(row["mean_of_problem_means"], 3.0)
        self.assertEqual(row["median"], 3.0)


if __name__ == "__main__":
    unittest.main()
