"""Find the patient's name in a medical document: a simple baseline, without machine learning.

The idea:
1. Check the document: Pydantic refuses a JSON that does not follow the challenge's format
   (a missing field, a coordinate outside [0, 1], a box whose minimum is above its maximum).
2. Put the words of each page in reading order: line by line, each line from left to right.
3. Rule 1: if the page is a form ("Nom: Hugo", "Prénom: Victor"), take the words written after the labels.
4. Rule 2: otherwise, find the first "Monsieur", "Madame"... that does not come right after "Dr" or
   "Docteur", and take the one or two words after it that start with a capital letter.

Usage from a terminal:
    python heuristic.py examples/example3.json
"""

import sys
from pathlib import Path
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# ---------------------------------------------------------------------------
# The challenge's JSON format, checked by Pydantic.
# ---------------------------------------------------------------------------


class BBox(BaseModel):
    """The box of a word. Coordinates go from 0 to 1, from the top-left corner of the page."""

    model_config = ConfigDict(frozen=True)

    x_min: float = Field(ge=0, le=1)
    x_max: float = Field(ge=0, le=1)
    y_min: float = Field(ge=0, le=1)
    y_max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def check_corners(self) -> Self:
        """Refuse a box whose minimum is after its maximum: the coordinates would be swapped."""
        if self.x_min > self.x_max or self.y_min > self.y_max:
            raise ValueError("x_min must not be above x_max, and y_min not above y_max")
        return self


class Word(BaseModel):
    """One word read on the page, with its box."""

    model_config = ConfigDict(frozen=True)

    text: str
    bbox: BBox


class Page(BaseModel):
    """One page: its words, in any order."""

    model_config = ConfigDict(frozen=True)

    words: list[Word]


class Document(BaseModel):
    """A whole document, as given by the challenge. Other top-level fields are ignored."""

    model_config = ConfigDict(frozen=True)

    pages: list[Page]


class PatientName(BaseModel):
    """The answer: the patient's first name (None when it is not written) and last name."""

    model_config = ConfigDict(frozen=True)

    first_name: str | None
    last_name: str


# ---------------------------------------------------------------------------
# Word lists and distances.
# ---------------------------------------------------------------------------

# Words that come just before the name of a person: "Monsieur Jean DUPONT", "Mme HUGO".
CIVILITIES: Final = frozenset({"monsieur", "madame", "mademoiselle", "m.", "mme", "mlle"})

# Words that come just before the name of a doctor.
# In "le docteur Monsieur Nicolas JACQUES", Nicolas JACQUES is the doctor, not the patient.
DOCTOR_TITLES: Final = frozenset({"dr", "dr.", "docteur", "pr", "pr.", "professeur", "docteure"})

# Coordinates go from 0 to 1 across the page.
# Two words whose tops differ by more than this (0.5% of the page height) are on two different lines.
LINE_GAP: Final = 0.005


# ---------------------------------------------------------------------------
# The heuristic.
# ---------------------------------------------------------------------------


def load_document(path: str | Path) -> Document:
    """Read a JSON file in the challenge's format and check it.

    Args:
        path: the JSON file.

    Returns:
        The checked document.

    Raises:
        pydantic.ValidationError: the file is not valid JSON, or does not follow the format.
    """
    return Document.model_validate_json(Path(path).read_text(encoding="utf-8"))


def read_words(page: Page) -> list[str]:
    """Return the words of a page in reading order: line by line, each line from left to right.

    Args:
        page: a page, with its words in any order.

    Returns:
        The texts of the words, in reading order.
    """
    # 1. Make a list of (y, x, text) for every word of the page.
    #    y is the top of the word and x its left side.
    words = [(word.bbox.y_min, word.bbox.x_min, word.text) for word in page.words]

    # 2. Sort the words from the top of the page to the bottom.
    #    Python sorts tuples by their first item (y), and by the second item (x) when y is equal.
    words.sort()

    # 3. Cut the sorted words into lines.
    #    The words of one line have almost the same y, so a new line starts only when y jumps
    #    by more than LINE_GAP. We cannot sort by (y, x) alone: on a scanned page, one word of a
    #    line can sit a tiny bit lower than the others and would be read in the wrong place.
    lines: list[list[tuple[float, str]]] = []
    previous_y = 0.0
    for y, x, text in words:
        if not lines or y - previous_y > LINE_GAP:
            lines.append([])  # start a new, empty line
        lines[-1].append((x, text))  # add the word to the current line (the last one of the list)
        previous_y = y

    # 4. Sort each line from left to right (by x), and put all the words in one simple list.
    result: list[str] = []
    for line in lines:
        line.sort()
        result.extend(text for x, text in line)
    return result


def find_patient(document: Document) -> PatientName | None:
    """Find the patient's name with two rules, tried in order on each page.

    Args:
        document: the checked document.

    Returns:
        The patient's name, or None when neither rule finds a name on any page.
    """
    for page in document.pages:
        words = read_words(page)

        # Rule 1: a form, like "Nom: Hugo" and "Prénom: Victor".
        # Look at each word. When it is "Nom:" or "Prénom:", keep the word that comes right after it.
        # We stop one word before the end of the list, because the last word has no word after it.
        first_name: str | None = None
        last_name: str | None = None
        for i in range(len(words) - 1):
            word = words[i].lower()
            if word == "nom:":
                last_name = words[i + 1]
            if word == "prénom:":
                first_name = words[i + 1]
        if last_name is not None:
            return PatientName(first_name=first_name, last_name=last_name)

        # Rule 2: the first civility ("Monsieur", "Madame"...) that is not right after a doctor title.
        for i, word in enumerate(words):
            if word.lower() not in CIVILITIES:
                continue  # not a civility: go to the next word

            # "le Docteur Monsieur Nicolas JACQUES": the word before is a doctor title,
            # so this name is the doctor's. Skip it and keep looking.
            if i >= 1 and words[i - 1].lower() in DOCTOR_TITLES:
                continue

            # Read at most two words after the civility, as long as they start with a capital letter.
            # "Monsieur Jean DUPONT pour" gives ["Jean", "DUPONT"]; "Mme HUGO un jour" gives ["HUGO"].
            name: list[str] = []
            for following in words[i + 1 : i + 3]:
                if following.lower() in DOCTOR_TITLES:
                    break  # "Monsieur le Docteur Nicolas JACQUES"
                # [:1] is the first letter, or "" for an empty word: an empty word stops the name.
                if not following[:1].isupper():
                    break  # a word in lowercase: the name is over ("Monsieur le Docteur ..." stops at "le")
                name.append(following.strip(".,"))  # "DUPONT," -> "DUPONT"

            # One word is a last name ("Mme HUGO"); two words are a first name and a last name ("Jean DUPONT").
            if len(name) == 1:
                return PatientName(first_name=None, last_name=name[0])
            if len(name) == 2:
                return PatientName(first_name=name[0], last_name=name[1])
            # No capitalized word after the civility (for example "Madame, Monsieur,"): keep looking.

    # Neither rule found a name on any page.
    return None


def main(argv: list[str] | None = None) -> int:
    """Print the patient's name for each JSON file given on the command line.

    Args:
        argv: the file paths; by default, the command-line arguments.

    Returns:
        0 when every file was read, 1 when at least one file is not in the challenge's format.
    """
    paths = sys.argv[1:] if argv is None else argv
    status = 0
    for path in paths:
        try:
            document = load_document(path)
        except ValidationError as error:
            print(f"{path}: invalid document\n{error}", file=sys.stderr)
            status = 1
            continue
        patient = find_patient(document)
        print(f"{path}: {patient.model_dump_json() if patient else 'null'}")
    return status


if __name__ == "__main__":
    sys.exit(main())
