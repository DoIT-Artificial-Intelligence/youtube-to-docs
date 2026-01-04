import unittest

import polars as pl

from youtube_to_docs import utils


class TestReorderColumns(unittest.TestCase):
    def test_reorder_columns_basic(self):
        # Create a DataFrame with jumbled columns
        data = {
            "Description": ["desc"],
            "URL": ["url"],
            "Tags": ["tags"],
            "Title": ["title"],
            "Extra": ["extra"],
        }
        df = pl.DataFrame(data)

        reordered = utils.reorder_columns(df)
        cols = reordered.columns

        # Check that known columns are at the start in correct order
        self.assertEqual(cols[0], "Title")
        self.assertEqual(cols[1], "URL")
        self.assertEqual(cols[2], "Description")
        self.assertEqual(cols[3], "Tags")

        # Check that unknown columns are at the end
        self.assertEqual(cols[-1], "Extra")

    def test_reorder_columns_complex(self):
        # Test with various dynamic columns
        cols = [
            "Summary File A",
            "Transcript File B",
            "URL",
            "Gemini STT cost",
            "Gemini Speaker extraction cost ($)",
            "Video File",
            "Random",
        ]
        # Create dummy df
        df = pl.DataFrame({c: [] for c in cols})

        reordered = utils.reorder_columns(df)
        final_cols = reordered.columns

        self.assertEqual(final_cols[0], "URL")

        # Verify logical groupings
        # "Transcript File B" should come before "STT cost" (based on function logic)

        # The function logic is:
        # 1. Base order (URL)
        # 2. Transcript chars
        # 3. Transcript Files (Transcript File B)
        # 4. STT costs (Gemini STT cost)
        # 5. Summary Files (Summary File A)
        # 6. Infographic Files
        # 7. Infographic Costs
        # 8. Audio Files
        # 9. Video Files (Video File)
        # 10. QA Files
        # 11. Speakers
        # 12. Speakers Files
        # 13. Speaker Costs (Gemini Speaker extraction cost ($))
        # ...

        expected_order_subset = [
            "URL",
            "Transcript File B",
            "Summary File A",
            "Video File",
            "Gemini STT cost",
            "Gemini Speaker extraction cost ($)",
            "Random",
        ]
        self.assertEqual(final_cols, expected_order_subset)


class TestModelNormalization(unittest.TestCase):
    def test_prefixes(self):
        self.assertEqual(utils.normalize_model_name("vertex-claude-3-5"), "claude-3-5")
        self.assertEqual(
            utils.normalize_model_name("bedrock-nova-2-lite"), "nova-2-lite"
        )
        self.assertEqual(utils.normalize_model_name("foundry-gpt-4"), "gpt-4")
        self.assertEqual(utils.normalize_model_name("claude-3-5"), "claude-3-5")

    def test_suffixes(self):
        self.assertEqual(
            utils.normalize_model_name("claude-haiku-4-5@20251001"), "claude-haiku-4-5"
        )
        self.assertEqual(
            utils.normalize_model_name("nova-2-lite-20251001-v1"), "nova-2-lite"
        )
        self.assertEqual(utils.normalize_model_name("model-v1"), "model")
        self.assertEqual(utils.normalize_model_name("model-v23"), "model")

    def test_combined(self):
        self.assertEqual(
            utils.normalize_model_name("bedrock-nova-2-lite-20251001-v1"), "nova-2-lite"
        )
        self.assertEqual(
            utils.normalize_model_name("vertex-claude-haiku-4-5@20251001"),
            "claude-haiku-4-5",
        )

    def test_unaffected(self):
        self.assertEqual(utils.normalize_model_name("gpt-4o"), "gpt-4o")
        self.assertEqual(
            utils.normalize_model_name("claude-3-5-sonnet"), "claude-3-5-sonnet"
        )


class TestQAProcessing(unittest.TestCase):
    def test_add_question_numbers_basic(self):
        input_table = (
            "| Who | What | Who | Ans |\n"
            "|---|---|---|---|\n"
            "| Me | Q1 | You | A1 |\n"
            "| Him | Q2 | Her | A2 |"
        )
        expected = (
            "| question number | Who | What | Who | Ans |\n"
            "|---|---|---|---|---|\n"
            "| 1 | Me | Q1 | You | A1 |\n"
            "| 2 | Him | Q2 | Her | A2 |"
        )
        self.assertEqual(utils.add_question_numbers(input_table), expected)

    def test_add_question_numbers_no_pipe_start(self):
        # Some LLMs might omit the starting pipe
        input_table = "Who | What | Who | Ans |\n---|---|---|---|\nMe | Q1 | You | A1 |"
        # Implementation adds pipe if missing for header/separator, and assumes
        # pipe for data rows check
        # Header: prepends "| question number |"
        expected = (
            "| question number |Who | What | Who | Ans |\n"
            "|---|---|---|---|---|\n"
            "| 1 | Me | Q1 | You | A1 |"
        )

        self.assertEqual(utils.add_question_numbers(input_table), expected)

    def test_add_question_numbers_empty(self):
        self.assertEqual(utils.add_question_numbers(""), "")

    def test_add_question_numbers_nan(self):
        # Nan strings should be handled before calling this, but if passed:
        # It won't have 2 lines, so returns as is.
        self.assertEqual(utils.add_question_numbers("nan"), "nan")
