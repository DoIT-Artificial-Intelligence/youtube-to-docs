import os
import tempfile
import unittest
import wave
from unittest.mock import MagicMock, patch

from youtube_to_docs import audio_description as ad
from youtube_to_docs.captions import CaptionFiles, Cue, srt_to_webvtt


def description(time, until, priority, text, duration=2.0, start=-1.0):
    d = ad.Description(time, until, priority, text, spoken=text)
    d.duration, d.start = duration, start
    return d


class TestMoments(unittest.TestCase):
    def test_find_moments_merges_bursts_and_adds_events(self):
        moments = ad.find_moments(
            [10.0, 11.5, 30.0],
            [(31.0, "Slide: Agenda"), (80.0, "Vote by raised hands")],
            duration=100.0,
        )
        self.assertEqual([m.time for m in moments], [0.0, 10.0, 30.0, 80.0])
        self.assertEqual(moments[1].until, 30.0)
        self.assertEqual(moments[-1].until, 100.0)
        self.assertEqual(moments[2].hint, "Slide: Agenda")
        self.assertEqual(moments[3].hint, "Vote by raised hands")

    def test_find_moments_caps_but_keeps_events(self):
        scenes = [float(t) for t in range(5, 5000, 5)]
        with patch.object(ad, "MAX_MOMENTS", 50):
            moments = ad.find_moments(scenes, [(2502.0, "Vote")], 5000.0)
        self.assertLessEqual(len(moments), 51)
        self.assertTrue(any(m.hint == "Vote" for m in moments))

    @patch("youtube_to_docs.audio_description._ffmpeg", return_value="ffmpeg")
    @patch("youtube_to_docs.audio_description.subprocess.run")
    def test_detect_scene_changes(self, mock_run, mock_ffmpeg):
        mock_run.return_value = MagicMock(
            returncode=0,
            stderr="[Parsed_showinfo_2] n:0 pts:1 pts_time:12.5 \\n"
            "[Parsed_showinfo_2] n:1 pts:2 pts_time:61.04 \\n",
        )
        self.assertEqual(ad.detect_scene_changes("v.mp4"), [12.5, 61.04])
        video_filter = mock_run.call_args.args[0][
            mock_run.call_args.args[0].index("-vf") + 1
        ]
        self.assertIn("select='gt(scene,0.3)'", video_filter)


class TestPlacement(unittest.TestCase):
    def test_speech_pauses(self):
        source = [Cue(0.0, 5.0, "a"), Cue(12.0, 20.0, "b")]
        reference = [
            Cue(5.4, 6.0, "jitter at the edge"),
            Cue(8.0, 9.0, "speech the first transcript missed"),
        ]
        pauses = ad.speech_pauses(source, reference, duration=30.0)
        self.assertEqual(
            [(round(a, 2), round(b, 2)) for a, b in pauses],
            [(5.25, 7.75), (9.25, 11.75), (20.25, 30.0)],
        )

    def test_place_descriptions_priority_room_and_staleness(self):
        pauses = [(10.0, 13.0), (20.0, 21.5), (40.0, 50.0)]
        slide = description(9.0, 35.0, 1, "Slide: Agenda.", duration=2.5)
        detail = description(9.5, 35.0, 2, "A logo in the corner.", duration=1.0)
        stale = description(15.0, 19.0, 1, "Replaced before any pause.", 1.0)
        late = description(18.0, 60.0, 1, "Too long for the next pause.", 4.0)
        ad.place_descriptions([detail, slide, stale, late], pauses)
        self.assertEqual(slide.start, 10.0)  # priority 1 first
        self.assertFalse(detail.placed)  # no room left in its window
        self.assertFalse(stale.placed)
        self.assertEqual(late.start, 40.0)

    def test_pauses_with_room(self):
        placed = description(10.0, 30.0, 1, "x", duration=2.0, start=10.0)
        free = ad.pauses_with_room([(10.0, 15.0), (20.0, 20.5)], [placed])
        self.assertEqual(free, [(10.0 + 2.0 + ad.NARRATION_GAP_SECONDS, 15.0)])


class TestOutputs(unittest.TestCase):
    def test_descriptions_webvtt(self):
        vtt = ad.descriptions_webvtt(
            [
                description(5, 9, 1, "Slide: Q&A <live>", duration=1.5, start=6.0),
                description(5, 9, 2, "Not placed", duration=1.0),
            ]
        )
        self.assertEqual(
            vtt,
            "WEBVTT\n\n00:00:06.000 --> 00:00:07.500\nSlide: Q&amp;A &lt;live>\n",
        )

    def test_descriptive_transcript(self):
        srt = (
            "1\n00:00:01,000 --> 00:00:03,000\n[Jane Doe] Welcome,\neveryone.\n\n"
            "2\n00:00:04,000 --> 00:00:06,000\nNext slide, please.\n\n"
            "3\n00:00:09,000 --> 00:00:10,000\n[Sam Lee] Thanks.\n"
        )
        text = ad.descriptive_transcript(
            srt,
            [description(5.0, 9, 1, 'Slide: "Agenda", three items.')],
            "Meeting",
        )
        self.assertIn("# Meeting: descriptive transcript", text)
        self.assertIn(
            "**[00:00:01] Jane Doe:** Welcome, everyone. Next slide, please.", text
        )
        self.assertIn('*[00:00:05] Description: Slide: "Agenda", three items.*', text)
        self.assertLess(text.index("Next slide"), text.index("Description"))
        self.assertLess(text.index("Description"), text.index("Sam Lee"))

    def test_descriptive_transcript_waits_for_the_sentence_to_end(self):
        srt = (
            "1\n00:00:01,000 --> 00:00:03,000\n[Jane Doe] So Dr.\n\n"
            "2\n00:00:03,000 --> 00:00:05,000\nThompson, welcome to\n\n"
            "3\n00:00:05,000 --> 00:00:07,000\nthe meeting. Thank you.\n"
        )
        text = ad.descriptive_transcript(
            srt, [description(2.5, 9, 1, "Slide: Agenda.")], "Meeting"
        )
        self.assertIn(
            "**[00:00:01] Jane Doe:** So Dr. Thompson, welcome to the meeting. "
            "Thank you.\n\n*[00:00:02] Description: Slide: Agenda.*",
            text,
        )

    def test_write_narration_track(self):
        clip = b"\x01\x00" * 100  # 100 samples
        placed = description(0, 9, 1, "x", duration=0.1, start=0.5)
        placed.audio, placed.rate = clip, 1000
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "n.wav")
            self.assertTrue(ad.write_narration_track([placed], 2.0, path))
            with wave.open(path, "rb") as track:
                self.assertEqual(track.getframerate(), 1000)
                self.assertEqual(track.getnframes(), 2000)
                frames = track.readframes(2000)
        self.assertEqual(frames[1000:1200], clip)
        self.assertEqual(frames[:1000], b"\x00" * 1000)

    def test_extended_descriptions_webvtt(self):
        vtt = ad.extended_descriptions_webvtt(
            [
                description(30.0, 60, 1, "Second slide."),
                description(10.0, 30, 1, "A long first description " * 4),
            ]
        )
        cues = [block.split("\n")[0] for block in vtt.split("\n\n")[1:]]
        self.assertEqual(cues[0], "00:00:10.000 --> 00:00:16.154")  # 16 words
        self.assertTrue(cues[1].startswith("00:00:30.000 --> "))

    def test_speakable_and_reading_the_track_back(self):
        self.assertEqual(
            ad.speakable("Slide: THRIVE 21613 FEEDER PATTERN, AIB and MCAP."),
            "Slide: Thrive 21613 Feeder Pattern, AIB and MCAP.",
        )
        found = ad.descriptions_from_webvtt(
            "WEBVTT\n\n00:00:05.000 --> 00:00:07.500\nSlide: Q&amp;A.\n"
        )
        self.assertEqual(
            [(d.start, d.duration, d.spoken) for d in found],
            [(5.0, 2.5, "Slide: Q&A.")],
        )

    def test_srt_to_webvtt(self):
        vtt = srt_to_webvtt(
            "1\n00:00:01,000 --> 00:00:02,500\n[Jane Doe]\nA & B <here>\n\n"
            "2\n00:01:00,000 --> 00:01:01,000\nNext.\n"
        )
        self.assertEqual(
            vtt,
            "WEBVTT\n\n00:00:01.000 --> 00:00:02.500\n[Jane Doe]\nA &amp; B &lt;here>"
            "\n\n00:01:00.000 --> 00:01:01.000\nNext.\n",
        )


class TestDescribe(unittest.TestCase):
    @patch("youtube_to_docs.audio_description._ask_with_images")
    def test_describe_moments_parses_answers(self, mock_ask):
        mock_ask.return_value = (
            "1|1|Title slide: AIB Public Meeting.|A title slide reads AIB Public "
            "Meeting, October 15.\n2|NONE\n3|2|A LOGO SPINS.\nnoise",
            10,
            5,
        )
        moments = [ad.Moment(0, 5), ad.Moment(5, 9), ad.Moment(9, 20, "Vote")]
        captions = [Cue(1, 2, "Welcome."), Cue(10, 11, "All in favor?")]
        with patch.object(ad, "FRAMES_PER_REQUEST", 8):
            found, in_tok, out_tok = ad.describe_moments(
                "gemini-4-argon", moments, ["", "", ""], captions, "Meeting", "- Jane"
            )
        self.assertEqual(
            [(d.time, d.priority, d.spoken, d.text) for d in found],
            [
                (
                    0,
                    1,
                    "Title slide: AIB Public Meeting.",
                    "A title slide reads AIB Public Meeting, October 15.",
                ),
                (9, 2, "A LOGO Spins.", "A LOGO SPINS."),
            ],
        )
        self.assertEqual((in_tok, out_tok), (10, 5))
        texts = " ".join(
            p.text for p in mock_ask.call_args.args[1] if getattr(p, "text", None)
        )
        self.assertIn("Frame 3 — 00:00:09", texts)
        self.assertIn("Said around this time: Welcome. All in favor?", texts)
        self.assertIn("Noted on screen around here: Vote", texts)
        self.assertIn("DCMP Description Key", texts)


class TestProcessAudioDescription(unittest.TestCase):
    OUTPUTS = {
        "vtt/gemini-4-argon - vid - Title - descriptions.vtt",
        "vtt/gemini-4-argon - vid - Title - extended descriptions.vtt",
        "dt/gemini-4-argon - vid - Title - descriptive transcript.md",
    }

    def _run(self, model_name="gemini-4-argon", existing=()):
        storage = MagicMock()
        storage.exists.side_effect = lambda path: path in existing
        storage.read_text.side_effect = lambda path: (
            "1\n00:00:01,000 --> 00:00:03,000\n[Jane Doe] Hello.\n"
        )
        storage.write_text.side_effect = lambda path, text: path
        row = {}
        get_video = MagicMock(return_value="video.mp4")
        result = ad.process_audio_description(
            row=row,
            storage=storage,
            model_name=model_name,
            captions=CaptionFiles("cc.srt", "gemini-3.5-transcribe", "stt.srt", None),
            get_video=get_video,
            tts_arg=None,
            video_id="vid",
            video_title="Title",
            safe_title="Title",
            dirs={"vtt": "vtt", "descriptive_transcripts": "dt"},
            duration_seconds=30.0,
            verbose=False,
            vprint=lambda *a, **k: None,
        )
        return row, storage, get_video, result

    def test_skips_non_gemini_and_existing(self):
        _, _, get_video, result = self._run(model_name="bedrock-claude-opus-5-5")
        get_video.assert_not_called()
        self.assertIsNone(result)
        row, _, get_video, result = self._run(existing=self.OUTPUTS)
        get_video.assert_not_called()
        self.assertIsNone(result)
        self.assertEqual(len(row), 3)

    @patch("youtube_to_docs.audio_description.synthesize")
    @patch("youtube_to_docs.audio_description.describe_moments")
    @patch("youtube_to_docs.audio_description.extract_frames", return_value=["f"])
    @patch("youtube_to_docs.audio_description.detect_scene_changes", return_value=[])
    def test_writes_tracks_and_transcript(
        self, mock_scenes, mock_frames, mock_describe, mock_tts
    ):
        mock_describe.return_value = (
            [
                ad.Description(
                    0.0, 30.0, 1, "A title slide reads Meeting.", spoken="Title slide."
                )
            ],
            1,
            1,
        )

        def fake_tts(descriptions, narration):
            self.assertEqual(narration, ad.DEFAULT_NARRATION_TTS)
            for d in descriptions:
                d.duration, d.audio, d.rate = 0.5, b"\x00\x00", 24000

        mock_tts.side_effect = fake_tts
        row, storage, _, placed = self._run()
        self.assertEqual(
            sorted(row),
            [
                "Descriptions VTT File gemini-4-argon",
                "Descriptive Transcript File gemini-4-argon",
                "Extended Descriptions VTT File gemini-4-argon",
            ],
        )
        written = {c.args[0]: c.args[1] for c in storage.write_text.call_args_list}
        self.assertIn(
            "Title slide.",
            written["vtt/gemini-4-argon - vid - Title - descriptions.vtt"],
        )
        extended = written[
            "vtt/gemini-4-argon - vid - Title - extended descriptions.vtt"
        ]
        self.assertIn("A title slide reads Meeting.", extended)
        self.assertEqual([d.spoken for d in placed], ["Title slide."])


if __name__ == "__main__":
    unittest.main()
