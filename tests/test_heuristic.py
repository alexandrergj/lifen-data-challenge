"""Unit tests for heuristic.py: known documents, each rule on small built pages, and input checks.

Run them with:  pytest
"""

import json
import random
from pathlib import Path

import pytest
from pydantic import ValidationError

from heuristic import (
    BBox,
    Document,
    Page,
    PatientName,
    Word,
    find_patient,
    load_document,
    main,
    read_words,
)

ROOT = Path(__file__).resolve().parent.parent


def make_document(*pages: list[str]) -> Document:
    """Build a document from lines of text, one list of lines per page.

    Each line gets its own height, words are placed left to right, and the words of each page are
    shuffled, like in the challenge's JSON.
    """
    rng = random.Random(0)  # always the same shuffle, so a test never changes from one run to the next
    built: list[Page] = []
    for lines in pages:
        words: list[Word] = []
        for row, line in enumerate(lines):
            y = 0.05 + 0.03 * row
            x = 0.05
            for text in line.split():
                width = 0.006 * max(len(text), 1)
                words.append(Word(text=text, bbox=BBox(x_min=x, x_max=x + width, y_min=y, y_max=y + 0.015)))
                x += width + 0.01
        rng.shuffle(words)
        built.append(Page(words=words))
    return Document(pages=built)


def name(first_name: str | None, last_name: str) -> PatientName:
    return PatientName(first_name=first_name, last_name=last_name)


# ---------------------------------------------------------------------------
# Known documents
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("example.json", name("Jean", "DUPONT")),
        ("examples/example1.json", name("Juliette", "MARTIN")),
        ("examples/example2.json", name("Victor", "Hugo")),
        ("examples/example3.json", name("Jean", "DUPONT")),
    ],
)
def test_challenge_examples(path: str, expected: PatientName) -> None:
    assert find_patient(load_document(ROOT / path)) == expected


# Documents where this version is known to be wrong. They are marked "xfail": the test suite passes
# while they fail, and it stops passing the day an improvement fixes one of them (strict mode),
# so that the test is then moved to the list above.
@pytest.mark.parametrize(
    ("path", "expected"),
    [
        pytest.param("examples/issues/issue1_letterhead_civility.json", name("Paul", "MOREAU"),
                     marks=pytest.mark.xfail(strict=True, reason="the secretary's civility comes first")),
        pytest.param("examples/issues/issue2_hospital_sticker.json", name("Jean", "DUPONT"),
                     marks=pytest.mark.xfail(strict=True, reason="'DUPONT Jean, né le': no clue recognized")),
        pytest.param("examples/issues/issue3_identity_block.json", name("Camille", "LEROY"),
                     marks=pytest.mark.xfail(strict=True, reason="'Nom de naissance :' is not 'Nom:'")),
        pytest.param("examples/issues/issue4_relative_first.json", name("Pierre", "DURAND"),
                     marks=pytest.mark.xfail(strict=True, reason="a relative is named before the patient")),
    ],
)
def test_known_limitations(path: str, expected: PatientName) -> None:
    assert find_patient(load_document(ROOT / path)) == expected


# ---------------------------------------------------------------------------
# Reading order
# ---------------------------------------------------------------------------


def test_read_words_puts_shuffled_words_back_in_order() -> None:
    document = make_document(["Compte rendu", "J'ai revu Monsieur Jean DUPONT"])
    assert read_words(document.pages[0]) == ["Compte", "rendu", "J'ai", "revu", "Monsieur", "Jean", "DUPONT"]


def test_read_words_keeps_a_slightly_lower_word_on_its_line() -> None:
    # Document 1 of the challenge: "Juliette" is 0.0003 lower than its neighbours.
    page = Page(words=[
        Word(text="MARTIN", bbox=BBox(x_min=0.40, x_max=0.48, y_min=0.1017, y_max=0.11)),
        Word(text="Juliette", bbox=BBox(x_min=0.30, x_max=0.38, y_min=0.1020, y_max=0.11)),
        Word(text="Madame", bbox=BBox(x_min=0.20, x_max=0.28, y_min=0.1017, y_max=0.11)),
    ])
    assert read_words(page) == ["Madame", "Juliette", "MARTIN"]


# ---------------------------------------------------------------------------
# Rule 1: form labels
# ---------------------------------------------------------------------------


def test_rule_1_reads_the_form_labels() -> None:
    document = make_document(["Patient:", "Nom: Hugo", "Prénom: Victor"])
    assert find_patient(document) == name("Victor", "Hugo")


def test_rule_1_without_first_name() -> None:
    assert find_patient(make_document(["Nom: Hugo"])) == name(None, "Hugo")


def test_rule_1_comes_before_rule_2() -> None:
    document = make_document(["Madame Lucie MARTIN", "Nom: Hugo", "Prénom: Victor"])
    assert find_patient(document) == name("Victor", "Hugo")


def test_rule_1_needs_the_exact_label() -> None:
    # "Nom :" with a space is two words, so rule 1 does not apply (a known limitation).
    assert find_patient(make_document(["Nom : Hugo", "Prénom : Victor"])) is None


# ---------------------------------------------------------------------------
# Rule 2: civility
# ---------------------------------------------------------------------------


def test_rule_2_reads_two_capitalized_words() -> None:
    document = make_document(["J'ai revu en consultation Monsieur Jean DUPONT pour une douleur"])
    assert find_patient(document) == name("Jean", "DUPONT")


def test_rule_2_one_word_is_a_last_name() -> None:
    assert find_patient(make_document(["revoir Mme HUGO un jour"])) == name(None, "HUGO")


def test_rule_2_removes_a_comma_or_a_full_stop() -> None:
    assert find_patient(make_document(["Monsieur DUPONT, né le 12/03/1965"])) == name(None, "DUPONT")


def test_rule_2_skips_a_civility_after_a_doctor_title() -> None:
    document = make_document(["adressé par le docteur Monsieur Nicolas JACQUES", "pour Madame Lucie MARTIN"])
    assert find_patient(document) == name("Lucie", "MARTIN")


def test_rule_2_stops_at_a_lowercase_word() -> None:
    # "Monsieur le Docteur ...": "le" ends the name before it starts, so the next civility answers.
    document = make_document(["Monsieur le Docteur Nicolas JACQUES", "Madame Lucie MARTIN"])
    assert find_patient(document) == name("Lucie", "MARTIN")


def test_rule_2_stops_at_a_doctor_title_after_the_civility() -> None:
    document = make_document(["Madame Dr Irène ADLER", "Monsieur Paul MOREAU"])
    assert find_patient(document) == name("Paul", "MOREAU")


def test_rule_2_skips_a_civility_without_a_name() -> None:
    assert find_patient(make_document(["Madame la directrice,", "Mme Lucie MARTIN"])) == name("Lucie", "MARTIN")


# ---------------------------------------------------------------------------
# No answer, several pages, unusual input
# ---------------------------------------------------------------------------


def test_no_clue_gives_none() -> None:
    assert find_patient(make_document(["Compte rendu de consultation", "Patient revu ce jour"])) is None


def test_empty_document_gives_none() -> None:
    assert find_patient(Document(pages=[])) is None
    assert find_patient(Document(pages=[Page(words=[])])) is None


def test_the_name_can_be_on_a_later_page() -> None:
    document = make_document(["Compte rendu"], ["Madame Lucie MARTIN"])
    assert find_patient(document) == name("Lucie", "MARTIN")


def test_an_empty_word_does_not_crash() -> None:
    page = Page(words=[
        Word(text="Madame", bbox=BBox(x_min=0.1, x_max=0.2, y_min=0.1, y_max=0.12)),
        Word(text="", bbox=BBox(x_min=0.21, x_max=0.22, y_min=0.1, y_max=0.12)),
    ])
    assert find_patient(Document(pages=[page])) is None


# ---------------------------------------------------------------------------
# Input checks (Pydantic)
# ---------------------------------------------------------------------------


def test_extra_fields_of_the_challenge_are_accepted() -> None:
    raw = {"pages": [], "original_page_count": 1, "needs_ocr_case": "no_ocr"}
    assert Document.model_validate(raw).pages == []


def test_a_word_without_box_is_refused() -> None:
    with pytest.raises(ValidationError):
        Document.model_validate({"pages": [{"words": [{"text": "Monsieur"}]}]})


def test_pixel_coordinates_are_refused() -> None:
    # Boxes in pixels instead of [0, 1] would put every word on its own line without any error.
    with pytest.raises(ValidationError):
        BBox(x_min=120, x_max=180, y_min=40, y_max=52)


def test_swapped_corners_are_refused() -> None:
    with pytest.raises(ValidationError):
        BBox(x_min=0.5, x_max=0.4, y_min=0.1, y_max=0.2)


def test_load_document_refuses_a_file_that_is_not_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_document(path)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def test_main_prints_one_answer_per_file(capsys: pytest.CaptureFixture[str]) -> None:
    status = main([str(ROOT / "examples/example3.json"), str(ROOT / "examples/issues/issue2_hospital_sticker.json")])
    lines = capsys.readouterr().out.splitlines()
    assert status == 0
    assert json.loads(lines[0].split(": ", 1)[1]) == {"first_name": "Jean", "last_name": "DUPONT"}
    assert lines[1].endswith(": null")


def test_main_reports_an_invalid_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "pixels.json"
    path.write_text(json.dumps({"pages": [{"words": [
        {"text": "Monsieur", "bbox": {"x_min": 120, "x_max": 180, "y_min": 40, "y_max": 52}},
    ]}]}), encoding="utf-8")
    assert main([str(path)]) == 1
    assert "invalid document" in capsys.readouterr().err
