import random
import unittest
from unittest.mock import MagicMock, patch

from youtube_to_docs import captions as cc


def srt(*cues):
    """SRT text from (start, end, text) tuples."""
    return "\n\n".join(
        f"{n}\n{cc.format_timestamp(s)} --> {cc.format_timestamp(e)}\n{t}"
        for n, (s, e, t) in enumerate(cues, 1)
    )


class TestParsing(unittest.TestCase):
    def test_parse_srt(self):
        text = (
            "1\r\n00:00:01,199 --> 00:00:05,838\r\n"
            "Good afternoon\r\nand welcome\r\n\r\n"
            "2\r\n01:02:03.5 --> 01:02:04,000\r\nSecond\r\n"
        )
        cues = cc.parse_srt(text)
        self.assertEqual(len(cues), 2)
        self.assertAlmostEqual(cues[0].start, 1.199)
        self.assertEqual(cues[0].text, "Good afternoon and welcome")
        self.assertAlmostEqual(cues[1].start, 3723.5)

    def test_parse_srt_from_google_doc(self):
        # Google Docs export each cue on one line, with a BOM and blank lines.
        text = (
            "\ufeff1 00:00:03,800 --> 00:00:04,100 [Speaker 1] Good morning.\r\n"
            "\r\n\r\n2 00:00:05,200 --> 00:00:06,500 My name\r\n"
        )
        cues = cc.parse_srt(text)
        self.assertEqual(
            [c.text for c in cues], ["[Speaker 1] Good morning.", "My name"]
        )
        self.assertAlmostEqual(cues[1].end, 6.5)

    def test_normalize_srt(self):
        doc = (
            "\ufeff1 00:00:01,000 --> 00:00:02,000 Hi\r\n\r\n\r\n"
            "2 00:00:03,000 --> 00:00:04,000 Bye\r\n"
        )
        self.assertEqual(
            cc.normalize_srt(doc),
            "1\n00:00:01,000 --> 00:00:02,000\nHi\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nBye\n",
        )
        well_formed = srt((1, 2, "Line one\nline two"), (3, 4, "Next"))
        self.assertEqual(cc.normalize_srt(well_formed), well_formed + "\n")

    def test_webvtt_from_google_doc_srt(self):
        doc = "\ufeff1 00:00:01,000 --> 00:00:02,500 Q&A <b>\r\n"
        self.assertEqual(
            cc.srt_to_webvtt(doc),
            "WEBVTT\n\n00:00:01.000 --> 00:00:02.500\nQ&amp;A &lt;b>\n",
        )

    def test_format_timestamp_rounds(self):
        self.assertEqual(cc.format_timestamp(3723.4999), "01:02:03,500")
        self.assertEqual(cc.format_timestamp(-1), "00:00:00,000")

    def test_prepare_moves_diarization_labels(self):
        cues = cc.prepare_cues(
            cc.parse_srt(
                srt(
                    (0, 1, "[Speaker 1] Hello there."),
                    (1, 2, "How are you?"),
                    (2, 3, "[Speaker 2] Fine."),
                )
            )
        )
        self.assertEqual(
            [c.label for c in cues], ["Speaker 1", "Speaker 1", "Speaker 2"]
        )
        self.assertEqual([c.turn_start for c in cues], [True, False, True])
        self.assertEqual(cues[0].text, "Hello there.")

    def test_prepare_derolls_and_splits_youtube_marks(self):
        # YouTube automatic captions overlap ("roll") and mark speakers with >>.
        cues = cc.prepare_cues(
            cc.parse_srt(
                srt(
                    (0, 4, "good afternoon and"),
                    (2, 6, "welcome to the"),
                    (4, 8, "board. >> Thank you"),
                    (6, 9, "so much."),
                )
            )
        )
        self.assertEqual(cues[0].end, 2)  # ends when the next line appears
        texts = [c.text for c in cues]
        self.assertEqual(texts[2:4], ["board.", "Thank you"])
        self.assertTrue(cues[3].turn_start)
        self.assertFalse(cues[2].turn_start)
        self.assertAlmostEqual(cues[2].end, cues[3].start)

    def test_srt_to_transcript_paragraph_per_speaker(self):
        content = (
            "1\n00:00:01,000 --> 00:00:02,000\n"
            "[Jane Doe] Good afternoon\nand welcome.\n\n"
            "2\n00:00:02,000 --> 00:00:03,000\nLet's begin.\n[John Roe] Thanks.\n\n"
            # One-line cue, as a Google Doc exports it.
            "3 00:00:03,000 --> 00:00:04,000 Sure.\n"
        )
        self.assertEqual(
            cc.srt_to_transcript(content),
            "[Jane Doe] Good afternoon and welcome. Let's begin.\n\n"
            "[John Roe] Thanks. Sure.\n",
        )
        self.assertEqual(cc.srt_to_transcript(""), "")


class TestAlignmentAndTiming(unittest.TestCase):
    def test_align_blocks_match_global_diff(self):
        rng = random.Random(1)
        words = [f"w{rng.randrange(800)}" for _ in range(6000)]
        edited = words[:2500] + ["extra", "words"] + words[2500:4000] + words[4100:]
        with patch.object(cc, "ALIGN_BLOCK_WORDS", 1000):
            pairs = cc._align(words, edited)
        self.assertGreater(len(pairs), 0.97 * len(edited))
        self.assertTrue(all(words[i] == edited[j] for i, j in pairs))
        self.assertEqual(pairs, sorted(pairs))

    def test_repair_from_reference(self):
        # Gemini STT occasionally returns words minutes off; the reference
        # transcript has the right times for the same words.
        cues = cc.prepare_cues(
            cc.parse_srt(
                srt(
                    (10, 12, "first cue is fine"),
                    (12, 1000, "this one ends far too late"),
                    (1000, 990, "and this starts late"),
                    (20, 22, "back to normal"),
                )
            )
        )
        reference = cc.prepare_cues(
            cc.parse_srt(
                srt(
                    (10, 12, "first cue is fine"),
                    (12, 15, "this one ends far too late"),
                    (15, 18, "and this starts late"),
                    (20, 22, "back to normal"),
                )
            )
        )
        cue_words, ref_words = cc._words(cues), cc._words(reference)
        pairs = cc._align([w.norm for w in cue_words], [w.norm for w in ref_words])
        self.assertEqual(cc.repair_timing(cues, cue_words, ref_words, pairs), 2)
        self.assertLess(abs(cues[1].start - 12), 0.5)
        self.assertLess(cues[1].end, 16)
        self.assertLess(abs(cues[2].start - 15), 0.5)
        self.assertLess(cues[2].end, 19)

    def test_repair_without_reference_interpolates(self):
        cues = cc.prepare_cues(
            cc.parse_srt(
                srt((10, 12, "one"), (12, 900, "two words"), (14, 16, "three"))
            )
        )
        self.assertEqual(cc.repair_timing(cues, cc._words(cues), [], []), 1)
        self.assertGreaterEqual(cues[1].start, 12)
        self.assertLessEqual(cues[1].end, 14)

    def test_reference_speaker_changes_become_hints(self):
        cues = cc.prepare_cues(
            cc.parse_srt(srt((0, 4, "Is that right? Yes it is."), (4, 6, "Good.")))
        )
        reference = cc.prepare_cues(
            cc.parse_srt(srt((0, 2, "is that right"), (2, 4, ">> yes it is good")))
        )
        cue_words, ref_words = cc._words(cues), cc._words(reference)
        pairs = cc._align([w.norm for w in cue_words], [w.norm for w in ref_words])
        cc.add_reference_hints(cues, cue_words, ref_words, pairs)
        self.assertEqual(cues[0].hints, [3])
        self.assertEqual(cc._hinted_text(cues[0]), "Is that right? >> Yes it is.")


class TestVideoContext(unittest.TestCase):
    def test_parse_video_context(self):
        context = cc.parse_video_context(
            "## People\n"
            "- Isiah Leggett (Chair, Accountability and Implementation Board (AIB)) "
            "— speaking at 00:01, 113:53\n"
            "- Kimberly Williams (Principal, Cecil County) — quoted on slide at 58:20\n"
            "\n## Names and terms\n- Blueprint for Maryland's Future\n"
            "\n## On-screen events\n- 14:22 — Vote board: 6 yes, 3 no\n"
        )
        leggett, williams = context.people
        self.assertEqual(leggett.name, "Isiah Leggett")
        self.assertEqual(
            leggett.title, "Chair, Accountability and Implementation Board (AIB)"
        )
        self.assertEqual(leggett.times, [1, 113 * 60 + 53])
        self.assertEqual(williams.note, "quoted on slide at 58:20")
        self.assertEqual(context.terms, ["Blueprint for Maryland's Future"])
        self.assertEqual(context.events, [(14 * 60 + 22, "Vote board: 6 yes, 3 no")])
        text = cc._people_text(context)
        self.assertIn("speaking at 00:00:01, 01:53:53", text)
        self.assertIn("Kimberly Williams (Principal, Cecil County) — quoted", text)


class TestSpeakerEvidence(unittest.TestCase):
    def test_video_times_attach_to_the_nearest_turn_start(self):
        cues = cc.prepare_cues(
            cc.parse_srt(
                srt(
                    (50.0, 52.0, "[Speaker 1] slides because there's a lot of data"),
                    (52.0, 53.0, "for you all. Liz."),
                    (55.0, 58.0, "[Speaker 2] So yes, there is a lot of synergy"),
                    (58.0, 60.0, "with MILE."),
                    (90.0, 92.0, "Later remarks."),
                )
            )
        )
        context = cc.parse_video_context(
            "## People\n- Liz Smith (Staff) — speaking at 00:53\n"
            "- Jane Doe (Chair) — speaking at 01:31\n"
            "- Absent Person (Guest) — speaking at 10:00\n"
        )
        hints = cc.video_speaker_hints(cues, context)
        # 00:53 is closest to the "for you all" cue, but Liz's turn starts at 55s.
        self.assertEqual(hints, {2: ["Liz Smith"], 4: ["Jane Doe"]})
        prompt = cc.build_speaker_prompt(
            cues, 0, len(cues), 1, 1, context, [], "Meeting", hints
        )
        self.assertIn("3 [00:00:55] (Speaker 2) {video: Liz Smith} So yes", prompt)
        self.assertIn("from cue 1 to cue 5", prompt)

    def test_speaker_windows_cover_every_cue(self):
        with patch.object(cc, "SPEAKER_WINDOW_CUES", 300):
            windows = cc._speaker_windows(2761)
        self.assertEqual(windows[0][0], 0)
        self.assertEqual(windows[-1][1], 2761)
        self.assertTrue(all(a[1] == b[0] for a, b in zip(windows, windows[1:])))
        self.assertTrue(all(e - s <= 300 for s, e in windows))


class TestResponseParsing(unittest.TestCase):
    def test_parse_speaker_lines(self):
        turns = cc.parse_speaker_lines(
            "1|Isiah Leggett\n#5 | Dr. Jane Smith (Presenter)\n7: [Speaker 3]\n"
            "notes about the task\n99|Out Of Range\n",
            range(1, 10),
        )
        self.assertEqual(
            turns,
            {1: "Isiah Leggett", 5: "Dr. Jane Smith", 7: cc.UNKNOWN_SPEAKER},
        )

    def test_parse_section_edits(self):
        speakers = ["A", "A", "B"]
        edits, additions = cc.parse_section_edits(
            "```\n1|A|Hello, everyone.\n2||>> [A] So we begin.\n"
            "2+|Chair|Thank you.\n3|B|Yes.\n3|C|No.\n4|D|context\n```",
            range(1, 4),
            speakers,
        )
        self.assertEqual(edits[1], [("A", "Hello, everyone.")])
        self.assertEqual(edits[2], [("A", "So we begin.")])
        self.assertEqual(edits[3], [("B", "Yes."), ("C", "No.")])
        self.assertEqual(additions, {2: [("Chair", "Thank you.")]})
        self.assertNotIn(4, edits)


class TestLayout(unittest.TestCase):
    def _captions(self, text, speaker="Jane Doe", start=0.0, end=None):
        end = end if end is not None else start + len(text) / 15
        segment = cc._Segment(start, end, speaker, cc._segment_tokens(start, end, text))
        return cc._captions_from_segments([segment])

    def test_limits_and_label(self):
        text = (
            "Let me thank all of you for attending today, and especially a thanks to "
            "our staff, who have struggled over the last several months to make sure "
            "that we were up and ready. In case there are some technical difficulties, "
            "we apologize in advance."
        )
        captions = self._captions(text)
        self.assertTrue(captions[0].lines[0].startswith("[Jane Doe] "))
        self.assertFalse(
            any(line.startswith("[") for c in captions[1:] for line in c.lines)
        )
        for caption in captions:
            self.assertLessEqual(len(caption.lines), 2)
            self.assertTrue(
                all(len(line) <= cc.MAX_LINE_CHARS for line in caption.lines)
            )
            self.assertLessEqual(
                caption.speech_end - caption.start, cc.MAX_CAPTION_SECONDS + 1e-6
            )
        joined = " ".join(" ".join(c.lines) for c in captions)
        self.assertEqual(joined.replace("[Jane Doe] ", ""), text)
        # Sentence ends are preferred caption ends.
        self.assertTrue(any(c.lines[-1].endswith("ready.") for c in captions))

    def test_em_dash_is_a_break_point(self):
        words = ["He's", "on", "the", "Eastern—", "he's", "at", "Salisbury,", "I"]
        words.append("believe.")
        spaces = [1, 1, 1, 1, 0, 1, 1, 1, 1]
        self.assertEqual(
            cc._layout_lines(words, spaces, ""),
            ["He's on the Eastern—", "he's at Salisbury, I believe."],
        )
        self.assertEqual(cc._join(["a—", "b", "c"], [1, 0, 1]), "a—b c")
        tokens = cc._segment_tokens(0, 2, "the Eastern—he's at")
        self.assertEqual([t.text for t in tokens], ["the", "Eastern—", "he's", "at"])
        self.assertEqual([t.glued for t in tokens], [False, False, True, False])

    def test_long_label_gets_its_own_line(self):
        words = "You said the difference?".split()
        self.assertEqual(
            cc._layout_lines(words, [1] * len(words), "[Dr. Jymil Thompson]"),
            ["[Dr. Jymil Thompson]", "You said the difference?"],
        )
        self.assertEqual(
            cc._layout_lines(["Yes."], [1], "[Dr. Jodi Colman]"),
            ["[Dr. Jodi Colman] Yes."],
        )

    def test_no_break_inside_names_or_after_articles(self):
        self.assertGreater(
            cc._pair_break_cost("Vice", "Chair"), cc._pair_break_cost("virtual,", "and")
        )
        self.assertGreater(
            cc._pair_break_cost("the", "board"), cc._pair_break_cost("board", "of")
        )
        self.assertGreater(
            cc._pair_break_cost("47", "percent"), cc._pair_break_cost("percent", "and")
        )
        self.assertEqual(
            cc._pair_break_cost("Dr.", "Smith"), cc._pair_break_cost("Dr.", "Smith")
        )
        self.assertFalse(cc._ends_sentence("Dr.", "Smith"))
        self.assertTrue(cc._ends_sentence("ready.", "In"))
        self.assertFalse(cc._ends_sentence("U.S.", "schools"))

    def test_crosstalk_moves_to_sentence_end(self):
        question = cc._Segment(
            0,
            6,
            "Rachel Hise",
            cc._segment_tokens(
                0, 6, "Is that the high school? It looks like the rate is 47%."
            ),
        )
        reply = cc._Segment(
            3.5, 3.8, "Dr. Thompson", cc._segment_tokens(3.5, 3.8, "Yes.")
        )
        ordered = cc._linearize([question, reply])
        texts = [" ".join(t.text for t in s.tokens) for s in ordered]
        self.assertEqual(
            texts,
            ["Is that the high school?", "Yes.", "It looks like the rate is 47%."],
        )
        self.assertTrue(
            all(a.end <= b.start + 1e-6 for a, b in zip(ordered, ordered[1:]))
        )

    def test_finalize_timing(self):
        captions = [
            cc._Caption(0.0, 0.2, 0.2, ["[A] Yes."], True),
            cc._Caption(
                0.3, 2.0, 2.0, ["[B] Then we can continue with the plan."], True
            ),
            cc._Caption(2.2, 3.0, 3.0, ["Okay."], False),
            cc._Caption(10.0, 10.5, 10.5, ["Later."], False),
        ]
        cc._finalize_timing(captions)
        self.assertGreaterEqual(
            captions[0].end - captions[0].start, cc.MIN_CAPTION_SECONDS - 1e-6
        )
        self.assertEqual(captions[1].end, captions[2].start)  # small gap closed
        self.assertTrue(all(a.end <= b.start for a, b in zip(captions, captions[1:])))
        self.assertLess(captions[2].end, 10.0)  # doesn't linger into silence

    def test_quick_exchange_becomes_two_line_caption(self):
        captions = [
            cc._Caption(0.0, 0.3, 0.3, ["[A] Ready?"], True),
            cc._Caption(0.4, 1.2, 1.2, ["[B] Yes, go ahead."], True),
        ]
        merged = cc._merge_quick_exchanges(captions)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].lines, ["[A] Ready?", "[B] Yes, go ahead."])


STT = srt(
    (1.0, 3.0, "[Speaker 1] okay good afternoon everyone"),
    (3.0, 6.0, "we will now adopt the minutes"),
    (7.0, 9.0, "[Speaker 2] so moved"),
    (9.5, 12.0, "[Speaker 1] all in favor"),
)
YOUTUBE = srt(
    (1.0, 4.0, "okay good afternoon everyone"),
    (3.0, 6.5, "we will now adopt the minutes"),
    (6.6, 7.0, ">> second"),
    (7.0, 9.4, ">> so moved"),
    (9.5, 12.0, ">> all in favor [laughter]"),
)


def fake_llm(model, prompt):
    if prompt.startswith("You are identifying"):
        return (
            "1|Isiah Leggett\n2|Isiah Leggett\n3|Rachel Hise (Executive Director)\n"
            "4|Isiah Leggett",
            10,
            2,
        )
    return (
        "1|Isiah Leggett|Okay, good afternoon, everyone.\n"
        "2|Isiah Leggett|We will now adopt the minutes.\n"
        "2+|Board Member|Second.\n"
        "3|Rachel Hise|So moved.\n"
        "4|Isiah Leggett|All in favor? [laughter]\n",
        20,
        5,
    )


class TestCorrectCaptions(unittest.TestCase):
    @patch.object(cc, "_query_llm", side_effect=fake_llm)
    def test_end_to_end(self, mock_llm):
        result = cc.correct_captions(
            "gemini-4-argon",
            STT,
            reference_srt=YOUTUBE,
            video_context="## People\n- Isiah Leggett (Chair) — speaking at 00:01",
            video_title="Meeting",
        )
        self.assertEqual(mock_llm.call_count, 2)
        self.assertEqual((result.input_tokens, result.output_tokens), (30, 7))
        self.assertEqual(result.added_segments, 1)
        self.assertEqual(result.fallback_cues, 0)
        cues = cc.parse_srt(result.srt)
        text = "\n".join(c.text for c in cues)
        self.assertIn("[Isiah Leggett] Okay, good afternoon, everyone.", text)
        self.assertIn("[Board Member] Second.", text)
        self.assertIn("[Rachel Hise] So moved.", text)
        self.assertIn("[Isiah Leggett] All in favor? [laughter]", text)
        self.assertNotIn("Speaker", text)
        # The added line is timed from where YouTube heard it.
        second = next(c for c in cues if "Second." in c.text)
        self.assertGreaterEqual(second.start, 6.0)
        self.assertLess(second.start, 7.0)
        self.assertTrue(all(a.end <= b.start for a, b in zip(cues, cues[1:])))
        prompts = [call.args[1] for call in mock_llm.call_args_list]
        self.assertIn("(Speaker 2)", prompts[0])
        self.assertIn(
            "1 [00:00:01] (Speaker 1) {video: Isiah Leggett} okay", prompts[0]
        )
        self.assertIn("WCAG 2.1", prompts[1])
        self.assertIn("Second transcript (YouTube captions)", prompts[1])
        self.assertIn("[00:00:06] >> second", prompts[1])

    @patch.object(cc, "_query_llm")
    @patch.object(cc.time, "sleep")
    def test_failed_sections_fall_back_to_source_text(self, mock_sleep, mock_llm):
        mock_llm.side_effect = lambda model, prompt: (
            ("1|Jane Doe\n2|Jane Doe\n3|Jane Doe\n4|Jane Doe", 1, 1)
            if prompt.startswith("You are identifying")
            else ("Error: rate limited", 0, 0)
        )
        result = cc.correct_captions("gemini-4-argon", STT)
        self.assertEqual(result.fallback_cues, 4)
        self.assertIn(
            "[Jane Doe] okay good afternoon everyone", result.srt.replace("\n", " ")
        )
        self.assertEqual(mock_llm.call_count, 1 + cc.LLM_ATTEMPTS)

    @patch.object(cc, "_query_llm")
    @patch.object(cc.time, "sleep")
    def test_ask_retries_until_valid(self, mock_sleep, mock_llm):
        mock_llm.side_effect = [("Error: 429", 0, 0), ("junk", 5, 1), ("1|A", 5, 1)]
        text, in_tok, out_tok = cc._ask("m", "p", lambda r: r.startswith("1|"))
        self.assertEqual((text, in_tok, out_tok), ("1|A", 10, 2))
        self.assertEqual(mock_sleep.call_count, 2)

    def test_empty_srt_raises(self):
        with self.assertRaises(ValueError):
            cc.correct_captions("m", "not an srt")


class TestProcessCorrectedCaptions(unittest.TestCase):
    def _run(self, row, source, transcript_arg, files, verbose=False, tokens=(100, 10)):
        storage = MagicMock()
        storage.exists.side_effect = lambda path: path in files
        storage.read_text.side_effect = lambda path: files[path]
        files = dict(files)

        def write(path, text):
            files[path] = text
            return path

        storage.write_text.side_effect = write
        with (
            patch.object(cc, "ensure_video_context", return_value="## People") as ctx,
            patch.object(cc, "correct_captions") as correct,
        ):
            correct.return_value = cc.CorrectedCaptions(
                "1\n00:00:01,000 --> 00:00:02,000\n[Jane Doe] Hi & bye.\n",
                *tokens,
                1,
                0,
                0,
                0,
            )
            cc.process_corrected_captions(
                row=row,
                storage=storage,
                model_name="gemini-4-argon",
                source=source,
                transcript_arg=transcript_arg,
                video_id="vid",
                video_title="Title",
                safe_title="Title",
                url="https://www.youtube.com/watch?v=vid",
                dirs={
                    "srt": "srt",
                    "transcript": "txt",
                    "vtt": "vtt",
                    "video_context": "ctx",
                },
                verbose=verbose,
                vprint=lambda *a, **k: None,
            )
        return correct, ctx, storage

    def test_corrects_requested_source_against_youtube(self):
        files = {
            "srt/gemini-3.5-transcribe generated - vid - Title.srt": "STT",
            "srt/youtube generated - vid - Title.srt": "YT",
        }
        row = {"Speakers gemini-4-argon from video-agentic": "Jane Doe (Chair)"}
        correct, ctx, storage = self._run(
            row, "gemini-3.5-transcribe", "gemini-3.5-transcribe", files
        )
        args, kwargs = correct.call_args
        self.assertEqual(args[:2], ("gemini-4-argon", "STT"))
        self.assertEqual(kwargs["reference_srt"], "YT")
        self.assertEqual(kwargs["video_context"], "## People")
        self.assertEqual(
            kwargs["speaker_lists"],
            [("gemini-4-argon from video-agentic", "Jane Doe (Chair)")],
        )
        self.assertEqual(
            row["Corrected Captions File gemini-4-argon from gemini-3.5-transcribe"],
            "srt/gemini-4-argon - vid - Title - "
            "corrected captions (from gemini-3.5-transcribe).srt",
        )
        # A WebVTT copy for websites (<track kind="captions">).
        vtt_path, vtt = storage.write_text.call_args_list[-1].args
        self.assertEqual(
            vtt_path,
            "vtt/gemini-4-argon - vid - Title - "
            "captions (from gemini-3.5-transcribe).vtt",
        )
        self.assertIn("[Jane Doe] Hi &amp; bye.", vtt)
        self.assertEqual(
            row["Captions VTT File gemini-4-argon from gemini-3.5-transcribe"], vtt_path
        )
        # And a plain-text transcript of the corrected captions.
        txt_path = (
            "txt/gemini-4-argon - vid - Title - "
            "corrected captions (from gemini-3.5-transcribe).txt"
        )
        self.assertEqual(
            row["Transcript File gemini-4-argon corrected from gemini-3.5-transcribe"],
            txt_path,
        )
        self.assertIn(
            (txt_path, "[Jane Doe] Hi & bye.\n"),
            [c.args for c in storage.write_text.call_args_list],
        )

    def test_missing_source_falls_back_to_youtube(self):
        files = {"srt/human generated - vid - Title.srt": "YT"}
        row = {}
        correct, _, _ = self._run(row, "gemini-3.5-transcribe", "youtube", files)
        self.assertEqual(correct.call_args.args[1], "YT")
        self.assertEqual(correct.call_args.kwargs["reference_srt"], "")
        self.assertIn("Corrected Captions File gemini-4-argon from youtube", row)

    def test_youtube_source_uses_stt_as_reference(self):
        files = {"stt.srt": "STT", "yt.srt": "YT"}
        row = {
            "SRT File youtube": "yt.srt",
            "SRT File gcp-chirp3": "stt.srt",
            "SRT File youtube (es)": "x",
        }
        correct, _, _ = self._run(row, "youtube", "youtube", files)
        self.assertEqual(correct.call_args.args[1], "YT")
        self.assertEqual(correct.call_args.kwargs["reference_srt"], "STT")

    def test_skips_when_output_exists(self):
        path = (
            "srt/gemini-4-argon - vid - Title - corrected captions (from youtube).srt"
        )
        files = {"srt/youtube generated - vid - Title.srt": "YT", path: "done"}
        row = {}
        correct, ctx, _ = self._run(row, "youtube", "youtube", files)
        correct.assert_not_called()
        ctx.assert_not_called()
        self.assertEqual(
            row["Corrected Captions File gemini-4-argon from youtube"], path
        )

    def test_existing_captions_get_missing_transcript(self):
        path = (
            "srt/gemini-4-argon - vid - Title - corrected captions (from youtube).srt"
        )
        files = {
            "srt/youtube generated - vid - Title.srt": "YT",
            path: "1\n00:00:01,000 --> 00:00:02,000\n[A] Hi\n",
        }
        row = {}
        correct, _, _ = self._run(row, "youtube", "youtube", files)
        correct.assert_not_called()
        self.assertEqual(
            row["Transcript File gemini-4-argon corrected from youtube"],
            "txt/gemini-4-argon - vid - Title - corrected captions (from youtube).txt",
        )

    def test_no_srt_does_nothing(self):
        correct, ctx, _ = self._run({}, None, "youtube", {})
        correct.assert_not_called()

    @patch.object(cc, "get_model_pricing", return_value=(2.0, 10.0))
    def test_verbose_records_cost(self, mock_pricing):
        files = {"srt/youtube generated - vid - Title.srt": "YT"}
        row = {}
        self._run(row, "youtube", "youtube", files, verbose=True, tokens=(10**6, 10**5))
        self.assertEqual(
            row["gemini-4-argon corrected captions cost from youtube ($)"], 3.0
        )


if __name__ == "__main__":
    unittest.main()
