import json

import pytest

from ml.inference.glossary import (
    GlossaryEntry,
    HeuristicEntityExtractor,
    NamedEntity,
    contains_term,
    load_glossary,
    merge_entities,
    missing_locked,
    propose_transliteration,
    relevant_locked,
    replace_literal_sources,
)


def test_ner_proposals_are_unlocked_and_user_terms_win():
    entities = HeuristicEntityExtractor().extract(["Alice met John Smith in London."])
    assert {entity.text for entity in entities} >= {"Alice", "John Smith", "London"}
    entries = merge_entities([GlossaryEntry("Alice", "ئەلیس", True, "person")], entities)
    assert next(item for item in entries if item.english == "Alice").sorani == "ئەلیس"
    assert all(not item.locked for item in entries if item.origin == "ner_proposal")
    assert propose_transliteration("John Smith") and propose_transliteration(
        "John Smith"
    ) == propose_transliteration("John Smith")
    categorized = merge_entities([], [NamedEntity("Paris", "place")])
    assert categorized[0].category == "place"
    normalized = merge_entities([GlossaryEntry("Kurd", "كورد")], [])
    assert normalized[0].sorani == "کورد"
    accented = HeuristicEntityExtractor().extract(["José visited Łódź and São Paulo."])
    assert {entity.text for entity in accented} == {"José", "Łódź", "São Paulo"}
    assert propose_transliteration("José") == propose_transliteration("Jose")
    assert not any(char.isascii() and char.isalpha() for char in propose_transliteration("Łódź"))


def test_locked_terms_are_whole_word_matched_and_literal_replacement_is_safe():
    terms = [GlossaryEntry("car", "سەیارە", True), GlossaryEntry("friend", "هاوڕێ", False)]
    assert not relevant_locked("The carpet is large", terms, "en-ckb")
    applicable = relevant_locked("Bring the CAR!", terms, "en-ckb")
    assert len(applicable) == 1
    assert missing_locked("Bring the car!", applicable, "en-ckb") == applicable
    replaced = replace_literal_sources("Bring the car!", applicable, "en-ckb")
    assert contains_term(replaced, "سەیارە") and "carpet" not in replaced
    assert (
        replace_literal_sources("No literal source here", applicable, "en-ckb")
        == "No literal source here"
    )
    assert relevant_locked("سەیارە لێرەیە", terms, "ckb-en") == applicable


def test_glossary_input_validation(tmp_path):
    path = tmp_path / "glossary.json"
    path.write_text(
        json.dumps([{"english": "Car", "sorani": "سەیارە", "locked": True}]), encoding="utf-8"
    )
    assert load_glossary(path)[0].locked
    path.write_text(
        json.dumps(
            [{"english": "Car", "sorani": "سەیارە"}, {"english": "car", "sorani": "ئۆتۆمبێل"}]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="more than once"):
        load_glossary(path)
