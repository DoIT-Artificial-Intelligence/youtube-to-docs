import unittest

from youtube_to_docs.post_process import post_process_transcript


class TestPostProcessTranscript(unittest.TestCase):
    def test_word_count_basic(self):
        transcript = "Apple banana apple cherry apple"
        result = post_process_transcript(transcript, '{"word count": "apple"}')
        self.assertEqual(result, {"Post-process: word count(apple)": 3})

    def test_word_count_case_insensitive(self):
        transcript = "HELLO hello Hello"
        result = post_process_transcript(transcript, '{"word count": "hello"}')
        self.assertEqual(result, {"Post-process: word count(hello)": 3})

    def test_word_count_list(self):
        transcript = "apple banana apple cherry banana banana"
        result = post_process_transcript(
            transcript, '{"word count": ["apple", "banana"]}'
        )
        self.assertEqual(result["Post-process: word count(apple)"], 2)
        self.assertEqual(result["Post-process: word count(banana)"], 3)

    def test_word_count_not_found(self):
        transcript = "apple banana cherry"
        result = post_process_transcript(transcript, '{"word count": "mango"}')
        self.assertEqual(result, {"Post-process: word count(mango)": 0})

    def test_word_count_whole_word(self):
        """Ensure partial matches inside words are not counted."""
        transcript = "pineapple apple applesauce"
        result = post_process_transcript(transcript, '{"word count": "apple"}')
        self.assertEqual(result, {"Post-process: word count(apple)": 1})

    def test_unknown_operation(self):
        transcript = "some text"
        result = post_process_transcript(transcript, '{"char count": "a"}')
        self.assertEqual(
            result, {"Post-process: unknown(char count)": "unsupported operation"}
        )

    def test_invalid_json(self):
        transcript = "some text"
        result = post_process_transcript(transcript, "not json")
        self.assertEqual(result, {})

    def test_empty_transcript(self):
        result = post_process_transcript("", '{"word count": "apple"}')
        self.assertEqual(result, {})

    def test_empty_json(self):
        result = post_process_transcript("hello world", "")
        self.assertEqual(result, {})

    def test_none_inputs(self):
        self.assertEqual(post_process_transcript(None, '{"word count": "a"}'), {})
        self.assertEqual(post_process_transcript("text", None), {})


if __name__ == "__main__":
    unittest.main()
