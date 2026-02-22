import unittest
from unittest.mock import patch

from youtube_to_docs.translator import parse_language_arg, translate_text


class TestTranslator(unittest.TestCase):
    def test_parse_language_arg(self):
        # Format: {model}-{language}
        self.assertEqual(parse_language_arg("gemini-es"), ("gemini", "es"))
        self.assertEqual(
            parse_language_arg("aws-translate-fr"), ("aws-translate", "fr")
        )
        self.assertEqual(parse_language_arg("google-cloud-de"), ("google-cloud", "de"))

        # Test with hyphens in the model name part
        self.assertEqual(
            parse_language_arg("gemini-3.1-flash-es"), ("gemini-3.1-flash", "es")
        )

        # Default/Fallback (though now we expect the hyphen format)
        self.assertEqual(parse_language_arg("es"), ("gemini", "es"))

    @patch("youtube_to_docs.translator._translate_aws")
    def test_translate_text_aws(self, mock_aws):
        mock_aws.return_value = "Hola Mundo"
        result, in_t, out_t = translate_text("Hello World", "aws-translate", "es")
        self.assertEqual(result, "Hola Mundo")
        self.assertEqual(in_t, 0)
        self.assertEqual(out_t, 0)
        mock_aws.assert_called_once_with("Hello World", "es", "en")

    @patch("youtube_to_docs.translator._translate_llm")
    def test_translate_text_llm(self, mock_llm):
        mock_llm.return_value = ("Hola Mundo", 10, 5)
        result, in_t, out_t = translate_text("Hello World", "gemini-flash", "es")
        self.assertEqual(result, "Hola Mundo")
        self.assertEqual(in_t, 10)
        self.assertEqual(out_t, 5)
        mock_llm.assert_called_once_with("Hello World", "gemini-flash", "es", "en")


if __name__ == "__main__":
    unittest.main()
