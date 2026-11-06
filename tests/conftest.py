import json
from pathlib import Path

import fasttext
import pytest

# Short, original sentences created solely as test fixtures; no movie data.
PARALLEL_LINES = [
    (
        ("The train is leaving the station.", "شەمەندەفەرەکە وێستگەکە بەجێدەهێڵێت."),
        ("We cannot leave before sunrise.", "ناتوانین پێش خۆرهەڵاتن بڕۆین."),
    ),
    (
        ("Please bring some water for me.", "تکایە هەندێک ئاو بۆ من بهێنە."),
        ("The garden needs a little care.", "باخچەکە پێویستی بە کەمێک چاودێرییە."),
    ),
    (
        ("We should cross the old bridge.", "دەبێت بەسەر پردە کۆنەکەدا بپەڕین."),
        ("A storm is coming from the west.", "زریانێک لە ڕۆژئاواوە دێت."),
    ),
    (
        ("My brother is waiting outside.", "براکەم لە دەرەوە چاوەڕێ دەکات."),
        ("I heard a knock on the window.", "گوێم لە لێدانی پەنجەرەکە بوو."),
    ),
    (
        ("This town is quiet at night.", "ئەم شارۆچکەیە لە شەودا ئارامە."),
        ("Our friends live near the mountain.", "هاوڕێکانمان لە نزیک شاخەکە دەژین."),
    ),
    (
        ("I found the key under the chair.", "کلیلەکەم لە ژێر کورسییەکە دۆزییەوە."),
        ("The music sounds beautiful tonight.", "مۆسیقاکە ئەم شەوە جوان دەنگ دەدات."),
    ),
]

LID_PAIRS = [
    ("hello friend how are you today", "سڵاو هاوڕێ چۆنیت ئەمڕۆ"),
    ("please open the door for us now", "تکایە دەرگاکە بۆ ئێمە بکەوە"),
    ("good morning and thank you friend", "بەیانی باش و سوپاس هاوڕێ"),
]
PERSIAN_TEXT = "سلام دوست من حالت چطور است امروز"


def write_srt(path: Path, lines: list[str]) -> None:
    blocks = []
    for index, text in enumerate(lines, 1):
        second = index * 3
        blocks.append(f"{index}\n00:00:{second:02},000 --> 00:00:{second + 2:02},000\n{text}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")


@pytest.fixture
def local_films(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    root.mkdir()
    for index, lines in enumerate(PARALLEL_LINES, 1):
        directory = root / f"film-{index}"
        directory.mkdir()
        write_srt(directory / "en.srt", [f"JOHN: {pair[0]}" for pair in lines])
        write_srt(directory / "ckb.srt", [pair[1] for pair in lines])
        (directory / "metadata.json").write_text(
            json.dumps({"provenance": "human", "quality_score": 0.9, "license": "test-only"}),
            encoding="utf-8",
        )
    return root


@pytest.fixture(scope="session")
def fasttext_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("lid")
    training = directory / "train.txt"
    examples = []
    for _ in range(25):
        for english, sorani in LID_PAIRS:
            examples.extend([f"__label__en {english}", f"__label__ckb {sorani}"])
        examples.append(f"__label__fa {PERSIAN_TEXT}")
    training.write_text("\n".join(examples) + "\n", encoding="utf-8")
    model = fasttext.train_supervised(
        input=str(training), epoch=40, lr=0.5, dim=16, minn=2, maxn=4, thread=1, verbose=0
    )
    path = directory / "lid.bin"
    model.save_model(str(path))
    return path
