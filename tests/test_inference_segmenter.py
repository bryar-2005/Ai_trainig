from ml.data.loaders import Cue
from ml.inference.segmenter import CueSpan, Sentence, reconstruct_sentences, split_translation


def cue(index, text, speaker=None):
    return Cue(
        index=index, start_ms=index * 1000, end_ms=index * 1000 + 900, text=text, speaker=speaker
    )


def test_reconstruction_offsets_ellipsis_speaker_and_empty_cues():
    cues = [
        cue(1, "Hello,", "JOHN"),
        cue(2, "my friend.", "JOHN"),
        cue(3, "Wait..."),
        cue(4, "we should go!"),
        cue(5, "Unfinished", "JANE"),
        cue(6, "New speaker.", "JOHN"),
        cue(7, ""),
        cue(8, "Final."),
    ]
    sentences = reconstruct_sentences(cues)
    assert [item.text for item in sentences] == [
        "Hello, my friend.",
        "Wait... we should go!",
        "Unfinished",
        "New speaker.",
        "Final.",
    ]
    assert sentences[0].cue_ids == (1, 2)
    for sentence in sentences:
        for span in sentence.spans:
            assert sentence.text[span.start : span.end] == cues[span.cue_index - 1].text


def test_speaker_change_across_unlabelled_continuation_flushes_the_group():
    cues = [cue(1, "We should", "JOHN"), cue(2, "leave now"), cue(3, "Stay here.", "JANE")]
    sentences = reconstruct_sentences(cues)
    assert [item.text for item in sentences] == ["We should leave now", "Stay here."]
    assert sentences[0].speaker == "JOHN" and sentences[1].speaker == "JANE"


def test_resplitting_never_loses_or_splits_words_even_when_cues_outnumber_words():
    for cue_count in range(1, 10):
        spans = tuple(
            CueSpan(index, index * 3, index * 3 + (index % 3 + 1)) for index in range(cue_count)
        )
        sentence = Sentence("source", spans, None)
        for word_count in range(16):
            words = [f"word{index}" for index in range(word_count)]
            parts = split_translation(sentence, " ".join(words))
            assert len(parts) == cue_count
            assert " ".join(part for part in parts if part) == " ".join(words)
            if word_count >= cue_count:
                assert all(parts)


def test_syntactic_split_stays_proportional_and_uses_whitespace_only():
    sentence = Sentence("A short source sentence here", (CueSpan(1, 0, 7), CueSpan(2, 8, 28)), None)
    result = split_translation(sentence, "one two, but three four five")
    assert len(result) == 2
    assert " ".join(result) == "one two, but three four five"
    assert result[0].endswith("two,")
