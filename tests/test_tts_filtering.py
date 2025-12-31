from unittest.mock import MagicMock, patch

import polars as pl

from youtube_to_docs.tts import process_tts


def test_tts_filtering():
    # Setup dummy data
    data = {
        "Summary File Model": ["dummy_en.md"],
        "Summary File Model (es)": ["dummy_es.md"],
        "Summary File Model (fr)": ["dummy_fr.md"],
    }
    df = pl.DataFrame(data)

    test_dir = "test_audio_out_pytest"

    # Mock Storage
    mock_storage = MagicMock()

    def read_text_side_effect(path):
        if "dummy_en.md" in path:
            return "Hello"
        if "dummy_es.md" in path:
            return "Hola"
        if "dummy_fr.md" in path:
            return "Bonjour"
        return ""

    mock_storage.read_text.side_effect = read_text_side_effect
    mock_storage.exists.return_value = False  # Assume audio doesn't exist
    mock_storage.write_bytes.return_value = "/saved/audio.wav"

    # Mock generate_speech to avoid API calls
    with patch("youtube_to_docs.tts.generate_speech") as mock_gen:
        mock_gen.return_value = b"fake_pcm_data"

        # --- Test 1: Only English ---
        df_en = process_tts(
            df, "test-model-voice", mock_storage, base_dir=test_dir, languages=["en"]
        )
        cols_en = df_en.columns

        assert "Summary Audio File Model test-model-voice File" in cols_en
        assert "Summary Audio File Model (es) test-model-voice File" not in cols_en
        assert "Summary Audio File Model (fr) test-model-voice File" not in cols_en

        # --- Test 2: Only Spanish ---
        df_es = process_tts(
            df, "test-model-voice", mock_storage, base_dir=test_dir, languages=["es"]
        )
        cols_es = df_es.columns

        assert "Summary Audio File Model test-model-voice File" not in cols_es
        assert "Summary Audio File Model (es) test-model-voice File" in cols_es
        assert "Summary Audio File Model (fr) test-model-voice File" not in cols_es

        # --- Test 3: English and French ---
        df_multi = process_tts(
            df,
            "test-model-voice",
            mock_storage,
            base_dir=test_dir,
            languages=["en", "fr"],
        )
        cols_multi = df_multi.columns

        assert "Summary Audio File Model test-model-voice File" in cols_multi
        assert "Summary Audio File Model (es) test-model-voice File" not in cols_multi
        assert "Summary Audio File Model (fr) test-model-voice File" in cols_multi

        # --- Test 4: No language filter (should process all) ---
        # Simulate languages=None (if that were passed) or implied all
        # But process_tts default is None, so we pass None
        df_all = process_tts(
            df, "test-model-voice", mock_storage, base_dir=test_dir, languages=None
        )
        cols_all = df_all.columns

        assert "Summary Audio File Model test-model-voice File" in cols_all
        assert "Summary Audio File Model (es) test-model-voice File" in cols_all
        assert "Summary Audio File Model (fr) test-model-voice File" in cols_all
