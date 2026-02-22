import unittest
from unittest.mock import MagicMock, patch

from youtube_to_docs.translate import (
    _translate_aws,
    _translate_gcp,
    parse_translate_arg,
    process_translate,
    translate_text,
)


class TestParseTranslateArg(unittest.TestCase):
    def test_simple_model_and_lang(self):
        model, lang = parse_translate_arg("gemini-3-flash-preview-es")
        self.assertEqual(model, "gemini-3-flash-preview")
        self.assertEqual(lang, "es")

    def test_bedrock_model(self):
        model, lang = parse_translate_arg("bedrock-nova-2-lite-v1-fr")
        self.assertEqual(model, "bedrock-nova-2-lite-v1")
        self.assertEqual(lang, "fr")

    def test_short_model(self):
        model, lang = parse_translate_arg("mymodel-de")
        self.assertEqual(model, "mymodel")
        self.assertEqual(lang, "de")

    def test_various_languages(self):
        cases = [
            ("gemini-3-flash-preview-ja", "gemini-3-flash-preview", "ja"),
            ("gemini-3-flash-preview-ko", "gemini-3-flash-preview", "ko"),
            ("gemini-3-flash-preview-zh", "gemini-3-flash-preview", "zh"),
            ("gemini-3-flash-preview-pt", "gemini-3-flash-preview", "pt"),
        ]
        for arg, expected_model, expected_lang in cases:
            with self.subTest(arg=arg):
                model, lang = parse_translate_arg(arg)
                self.assertEqual(model, expected_model)
                self.assertEqual(lang, expected_lang)

    def test_aws_translate(self):
        model, lang = parse_translate_arg("aws-translate-es")
        self.assertEqual(model, "aws-translate")
        self.assertEqual(lang, "es")

    def test_aws_translate_fr(self):
        model, lang = parse_translate_arg("aws-translate-fr")
        self.assertEqual(model, "aws-translate")
        self.assertEqual(lang, "fr")

    def test_gcp_translate(self):
        model, lang = parse_translate_arg("gcp-translate-es")
        self.assertEqual(model, "gcp-translate")
        self.assertEqual(lang, "es")

    def test_gcp_translate_fr(self):
        model, lang = parse_translate_arg("gcp-translate-fr")
        self.assertEqual(model, "gcp-translate")
        self.assertEqual(lang, "fr")

    def test_invalid_no_dash(self):
        with self.assertRaises(ValueError):
            parse_translate_arg("nodashhere")

    def test_invalid_empty(self):
        with self.assertRaises(ValueError):
            parse_translate_arg("")


class TestTranslateAws(unittest.TestCase):
    @patch("youtube_to_docs.translate.boto3")
    def test_calls_aws_translate(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.translate_text.return_value = {"TranslatedText": "Hola mundo"}

        result, in_tok, out_tok = _translate_aws("Hello world", "es")

        self.assertEqual(result, "Hola mundo")
        self.assertEqual(in_tok, 0)
        self.assertEqual(out_tok, 0)
        mock_boto3.client.assert_called_once_with("translate", region_name="us-east-1")
        mock_client.translate_text.assert_called_once_with(
            Text="Hello world",
            SourceLanguageCode="en",
            TargetLanguageCode="es",
        )

    @patch("youtube_to_docs.translate.boto3")
    def test_uses_aws_region_env(self, mock_boto3):
        import os

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.translate_text.return_value = {"TranslatedText": "Bonjour"}

        with patch.dict(os.environ, {"AWS_REGION": "eu-west-1"}):
            _translate_aws("Hello", "fr")

        mock_boto3.client.assert_called_once_with("translate", region_name="eu-west-1")


class TestTranslateGcp(unittest.TestCase):
    @patch("youtube_to_docs.translate.google_translate")
    def test_calls_gcp_translate(self, mock_google_translate):
        mock_client = MagicMock()
        mock_google_translate.Client.return_value = mock_client
        mock_client.translate.return_value = {"translatedText": "Hola mundo"}

        result, in_tok, out_tok = _translate_gcp("Hello world", "es")

        self.assertEqual(result, "Hola mundo")
        self.assertEqual(in_tok, 0)
        self.assertEqual(out_tok, 0)
        mock_google_translate.Client.assert_called_once_with()
        mock_client.translate.assert_called_once_with(
            "Hello world",
            target_language="es",
            source_language="en",
            format_="text",
        )

    @patch("youtube_to_docs.translate.google_translate")
    def test_joins_translated_chunks(self, mock_google_translate):
        mock_client = MagicMock()
        mock_google_translate.Client.return_value = mock_client
        # Return different text per call to verify chunk joining
        mock_client.translate.side_effect = [
            {"translatedText": "Hola"},
            {"translatedText": " mundo"},
        ]

        # Force two chunks by using a tiny max_bytes
        from youtube_to_docs import translate as translate_mod

        original_limit = translate_mod._GCP_TRANSLATE_CHAR_LIMIT
        translate_mod._GCP_TRANSLATE_CHAR_LIMIT = 5
        try:
            result, _, _ = _translate_gcp("Hello\nworld", "es")
        finally:
            translate_mod._GCP_TRANSLATE_CHAR_LIMIT = original_limit

        self.assertEqual(result, "Hola mundo")

    @patch("youtube_to_docs.translate.google_translate", None)
    def test_returns_error_when_library_missing(self):
        result, in_tok, out_tok = _translate_gcp("Hello", "es")

        self.assertIn("google-cloud-translate", result)
        self.assertEqual(in_tok, 0)
        self.assertEqual(out_tok, 0)


class TestTranslateText(unittest.TestCase):
    @patch("youtube_to_docs.translate._query_llm")
    def test_calls_query_llm_with_correct_prompt(self, mock_query):
        mock_query.return_value = ("Hola mundo", 10, 5)

        result, in_tok, out_tok = translate_text(
            "gemini-3-flash-preview", "Hello world", "es"
        )

        self.assertEqual(result, "Hola mundo")
        self.assertEqual(in_tok, 10)
        self.assertEqual(out_tok, 5)

        prompt_used = mock_query.call_args[0][1]
        self.assertIn("es", prompt_used)
        self.assertIn("Hello world", prompt_used)

    @patch("youtube_to_docs.translate._query_llm")
    def test_target_language_in_prompt(self, mock_query):
        mock_query.return_value = ("Bonjour", 5, 3)

        translate_text("gemini-3-flash-preview", "Hello", "fr")

        prompt = mock_query.call_args[0][1]
        self.assertIn("fr", prompt)

    @patch("youtube_to_docs.translate._query_llm")
    def test_returns_query_llm_tuple(self, mock_query):
        mock_query.return_value = ("translated", 100, 50)

        result = translate_text("some-model", "some text", "de")

        self.assertEqual(result, ("translated", 100, 50))

    @patch("youtube_to_docs.translate._translate_aws")
    def test_routes_to_aws_translate(self, mock_aws):
        mock_aws.return_value = ("Hola", 0, 0)

        result = translate_text("aws-translate", "Hello", "es")

        self.assertEqual(result, ("Hola", 0, 0))
        mock_aws.assert_called_once_with("Hello", "es")

    @patch("youtube_to_docs.translate._translate_aws")
    @patch("youtube_to_docs.translate._query_llm")
    def test_does_not_call_llm_for_aws_translate(self, mock_query, mock_aws):
        mock_aws.return_value = ("Hola", 0, 0)

        translate_text("aws-translate", "Hello", "es")

        mock_query.assert_not_called()

    @patch("youtube_to_docs.translate._translate_gcp")
    def test_routes_to_gcp_translate(self, mock_gcp):
        mock_gcp.return_value = ("Hola", 0, 0)

        result = translate_text("gcp-translate", "Hello", "es")

        self.assertEqual(result, ("Hola", 0, 0))
        mock_gcp.assert_called_once_with("Hello", "es")

    @patch("youtube_to_docs.translate._translate_gcp")
    @patch("youtube_to_docs.translate._query_llm")
    def test_does_not_call_llm_for_gcp_translate(self, mock_query, mock_gcp):
        mock_gcp.return_value = ("Hola", 0, 0)

        translate_text("gcp-translate", "Hello", "es")

        mock_query.assert_not_called()


class TestProcessTranslate(unittest.TestCase):
    def _make_storage(self):
        storage = MagicMock()
        storage.write_text.side_effect = lambda path, text: path
        return storage

    def _base_row(self, model="gemini-3-flash-preview", transcript_arg="youtube"):
        return {
            f"Summary Text {model} from {transcript_arg}": "English summary",
            f"One Sentence Summary {model} from {transcript_arg}": "One sentence.",
            f"QA Text {model} from {transcript_arg}": "| Q | A |",
            f"Tags {transcript_arg} {model} model": "tag1, tag2",
        }

    @patch("youtube_to_docs.translate._query_llm")
    def test_translates_all_columns(self, mock_query):
        mock_query.return_value = ("traducido", 10, 5)
        model = "gemini-3-flash-preview"
        transcript_arg = "youtube"
        row = self._base_row(model, transcript_arg)

        result = process_translate(
            row=row,
            translate_model=model,
            translate_lang="es",
            transcript_arg=transcript_arg,
            model_names=[model],
            summaries_dir="summary-files",
            one_sentence_summaries_dir="one-sentence-summary-files",
            qa_dir="qa-files",
            tags_dir="tag-files",
            video_id="abc123",
            safe_title="My Video",
            storage=self._make_storage(),
        )

        self.assertEqual(
            result[f"Summary Text {model} from {transcript_arg} (es)"], "traducido"
        )
        self.assertEqual(
            result[f"One Sentence Summary {model} from {transcript_arg} (es)"],
            "traducido",
        )
        self.assertEqual(
            result[f"QA Text {model} from {transcript_arg} (es)"], "traducido"
        )
        self.assertEqual(
            result[f"Tags {transcript_arg} {model} model (es)"], "traducido"
        )

    @patch("youtube_to_docs.translate._query_llm")
    def test_skips_already_translated_columns(self, mock_query):
        mock_query.return_value = ("traducido", 10, 5)
        model = "gemini-3-flash-preview"
        transcript_arg = "youtube"
        row = self._base_row(model, transcript_arg)
        # Pre-populate the translated column
        row[f"Summary Text {model} from {transcript_arg} (es)"] = "already translated"

        result = process_translate(
            row=row,
            translate_model=model,
            translate_lang="es",
            transcript_arg=transcript_arg,
            model_names=[model],
            summaries_dir="summary-files",
            one_sentence_summaries_dir="one-sentence-summary-files",
            qa_dir="qa-files",
            tags_dir="tag-files",
            video_id="abc123",
            safe_title="My Video",
            storage=self._make_storage(),
        )

        # Should NOT have been overwritten
        self.assertEqual(
            result[f"Summary Text {model} from {transcript_arg} (es)"],
            "already translated",
        )

    @patch("youtube_to_docs.translate._query_llm")
    def test_skips_missing_english_columns(self, mock_query):
        mock_query.return_value = ("traducido", 10, 5)
        model = "gemini-3-flash-preview"
        transcript_arg = "youtube"
        # Row has no English content at all
        row = {}

        result = process_translate(
            row=row,
            translate_model=model,
            translate_lang="es",
            transcript_arg=transcript_arg,
            model_names=[model],
            summaries_dir="summary-files",
            one_sentence_summaries_dir="one-sentence-summary-files",
            qa_dir="qa-files",
            tags_dir="tag-files",
            video_id="abc123",
            safe_title="My Video",
            storage=self._make_storage(),
        )

        # No translated columns should be added
        self.assertNotIn(f"Summary Text {model} from {transcript_arg} (es)", result)
        mock_query.assert_not_called()

    @patch("youtube_to_docs.translate._query_llm")
    def test_translates_secondary_youtube_columns(self, mock_query):
        mock_query.return_value = ("traducido", 10, 5)
        model = "gemini-3-flash-preview"
        transcript_arg = "gemini-3-flash-preview"
        row = self._base_row(model, transcript_arg)
        # Add secondary "from youtube" columns
        row[f"Summary Text {model} from youtube"] = "YouTube English summary"
        row[f"One Sentence Summary {model} from youtube"] = "YouTube one sentence."
        row[f"QA Text {model} from youtube"] = "| YT Q | YT A |"

        result = process_translate(
            row=row,
            translate_model=model,
            translate_lang="fr",
            transcript_arg=transcript_arg,
            model_names=[model],
            summaries_dir="summary-files",
            one_sentence_summaries_dir="one-sentence-summary-files",
            qa_dir="qa-files",
            tags_dir="tag-files",
            video_id="abc123",
            safe_title="My Video",
            storage=self._make_storage(),
        )

        self.assertIn(f"Summary Text {model} from youtube (fr)", result)
        self.assertIn(f"One Sentence Summary {model} from youtube (fr)", result)
        self.assertIn(f"QA Text {model} from youtube (fr)", result)

    @patch("youtube_to_docs.translate._query_llm")
    def test_saves_translated_files(self, mock_query):
        mock_query.return_value = ("traducido", 10, 5)
        model = "gemini-3-flash-preview"
        transcript_arg = "youtube"
        row = self._base_row(model, transcript_arg)
        storage = self._make_storage()

        process_translate(
            row=row,
            translate_model=model,
            translate_lang="es",
            transcript_arg=transcript_arg,
            model_names=[model],
            summaries_dir="summary-files",
            one_sentence_summaries_dir="one-sentence-summary-files",
            qa_dir="qa-files",
            tags_dir="tag-files",
            video_id="abc123",
            safe_title="My Video",
            storage=storage,
        )

        # write_text should have been called for each translatable column
        self.assertEqual(storage.write_text.call_count, 4)

    @patch("youtube_to_docs.translate._query_llm")
    def test_multiple_models(self, mock_query):
        mock_query.return_value = ("traducido", 10, 5)
        model_a = "gemini-3-flash-preview"
        model_b = "bedrock-nova-2-lite-v1"
        transcript_arg = "youtube"
        row = {
            **self._base_row(model_a, transcript_arg),
            **self._base_row(model_b, transcript_arg),
        }

        result = process_translate(
            row=row,
            translate_model=model_a,
            translate_lang="de",
            transcript_arg=transcript_arg,
            model_names=[model_a, model_b],
            summaries_dir="summary-files",
            one_sentence_summaries_dir="one-sentence-summary-files",
            qa_dir="qa-files",
            tags_dir="tag-files",
            video_id="abc123",
            safe_title="My Video",
            storage=self._make_storage(),
        )

        for model in [model_a, model_b]:
            self.assertIn(f"Summary Text {model} from {transcript_arg} (de)", result)
            self.assertIn(f"QA Text {model} from {transcript_arg} (de)", result)

    @patch("youtube_to_docs.translate._query_llm")
    def test_file_col_stored_in_row(self, mock_query):
        mock_query.return_value = ("traducido", 10, 5)
        model = "gemini-3-flash-preview"
        transcript_arg = "youtube"
        row = self._base_row(model, transcript_arg)
        storage = self._make_storage()

        result = process_translate(
            row=row,
            translate_model=model,
            translate_lang="es",
            transcript_arg=transcript_arg,
            model_names=[model],
            summaries_dir="summary-files",
            one_sentence_summaries_dir="one-sentence-summary-files",
            qa_dir="qa-files",
            tags_dir="tag-files",
            video_id="abc123",
            safe_title="My Video",
            storage=storage,
        )

        # File path columns should be populated
        self.assertIn(f"Summary File {model} from {transcript_arg} (es)", result)
        self.assertIn(f"QA File {model} from {transcript_arg} (es)", result)
        self.assertIn(f"Tags File {transcript_arg} {model} model (es)", result)


if __name__ == "__main__":
    unittest.main()
