import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from online_cost_optimism import collect as original_collect
from online_early_start import collect
from two_reward_models import CONFIGS, load_pools


class ServerRunnerTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / 'input.jsonl'
        self.rows = [
            dict(JSON_idx=i, generations=[
                dict(text='response' * (j+1), token_count=j+1,
                     fsfairx_rm_reward=float(j+i), mistral_rm_reward=float(j-i))
                for j in range(8)])
            for i in range(4)
        ]
        self.write_rows()

    def write_rows(self):
        self.path.write_text('\n'.join(json.dumps(r) for r in self.rows) + '\n')

    def test_tokens_are_recorded_counts_not_text_lengths(self):
        pools = load_pools(self.path, 'fsfairx_rm_reward', 'token_count', 'tokens')
        np.testing.assert_array_equal(pools[0].lengths, np.arange(1, 9))
        chars = load_pools(self.path, 'fsfairx_rm_reward', 'text_chars', 'characters')
        np.testing.assert_array_equal(chars[0].lengths, 8*np.arange(1, 9))

    def test_characters_cannot_be_labeled_tokens(self):
        with self.assertRaisesRegex(ValueError, 'not tokens'):
            load_pools(self.path, 'fsfairx_rm_reward', 'text_chars', 'tokens')

    def test_missing_tokens_do_not_fall_back_to_characters(self):
        del self.rows[0]['generations'][0]['token_count']
        self.write_rows()
        with self.assertRaises(KeyError):
            load_pools(self.path, 'fsfairx_rm_reward', 'token_count', 'tokens')

    def test_invalid_counts_are_rejected(self):
        for count in [0, -1, 1.5, float('nan')]:
            with self.subTest(count=count):
                self.rows[0]['generations'][0]['token_count'] = count
                self.write_rows()
                with self.assertRaises(ValueError):
                    load_pools(self.path, 'fsfairx_rm_reward', 'token_count', 'tokens')

    def test_four_sample_rule_matches_original_replay(self):
        pools = load_pools(self.path, 'mistral_rm_reward', 'text_chars', 'characters')
        actual, fixed = collect(pools, 8, 20260923, [CONFIGS[-1]])
        expected, expected_fixed = original_collect(
            pools, 8, 8, 20260923, [dict(smoothing='mean', cost_optimism=2.)])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(fixed, expected_fixed)


if __name__ == '__main__':
    unittest.main()
