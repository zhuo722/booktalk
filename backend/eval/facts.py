"""The fixture book's fact table — invented on purpose (SPEC §10).

Every name, year, population and dish below is made up. That is the whole
point: an evaluation of *retrieval* has to be contamination-free, and a model
that has read the real world can answer "how tall is the Eiffel Tower" without
retrieving anything at all. Nothing in ``A Field Guide to the Meridian
Archipelago`` exists, so a correct answer is proof that the pipeline found the
passage, and a confident wrong answer is proof that it did not.

The table is the single source of truth for three artefacts, all generated from
it by :mod:`make_fixture_book`:

* the **prose** of the fixture PDF — every fact is woven into a sentence that
  the book actually prints;
* the **golden questions** — each one derived from the same row it is testing,
  so a question can never drift away from the book that answers it;
* the **evidence spans** — the exact sentences a perfect retriever would return,
  which is what the token-level IoU metric of SPEC §14.10 is measured against.

Comparisons across chapters (populations, charter years, distances) are computed
from the table at build time rather than written by hand, so "which island is
older" cannot disagree with the two chapters it is asking about.

The ``heard`` field on each island is the ASR-corruption arm: a phonetic
mis-hearing of the island's distinctive name, of the kind
``gpt-4o-mini-transcribe`` produces on a word it has never seen. It must survive
:func:`app.speech.phonetics.repair_transcript` — ``make_fixture_book`` asserts
that it does, against the vocabulary the real ingest path would extract, so a
corruption that stops being repairable fails fixture generation rather than
silently turning into an unpassable eval question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

BOOK_TITLE = "A Field Guide to the Meridian Archipelago"
BOOK_SUBTITLE = "Twelve Islands, Their Charters, Their Landmarks and Their Kitchens"
BOOK_AUTHOR = "Compiled for the Assembly at Ashgate Quay"
#: The mainland port every crossing in the book is measured from.
CAPITAL = "Ashgate Quay"


@dataclass(frozen=True)
class Island:
    """One chapter of the fixture book: an island and its eight facts."""

    idx: int
    name: str
    founder: str
    founded: int
    population: int
    landmark: str  # written with its article, e.g. "the Sennet Lantern"
    height_m: int
    dish: str
    ingredient: str
    distance_nm: int
    festival: str  # written with its article, e.g. "the Lantern Vigil"
    month: str
    export: str
    custom: str  # a whole sentence: the island's one local peculiarity
    terrain: str
    weather: str
    bird: str
    craft: str
    theme: str
    #: A phonetic mis-hearing of :attr:`key_term`, for the ASR-corruption arm.
    heard: str

    @property
    def key_term(self) -> str:
        """The distinctive first word of the island's name ("Kestrel")."""
        return self.name.split()[0]

    @property
    def landmark_name(self) -> str:
        """The landmark without its article ("Sennet Lantern")."""
        return _bare(self.landmark)

    @property
    def festival_name(self) -> str:
        """The festival without its article ("Lantern Vigil")."""
        return _bare(self.festival)

    @property
    def chapter_title(self) -> str:
        return "Chapter {0} — {1}".format(self.idx, self.name)

    @property
    def sections(self) -> Tuple[str, str]:
        """The two section headings inside the chapter.

        Island-specific rather than generic ("Settlement and charter" twelve
        times over) so that the breadcrumb a chunk carries into the index names
        a place, and so no two sections of the book share a heading path.
        """
        return (
            "Settlement of {0}".format(self.name),
            "Landmarks, table and trade of {0}".format(self.name),
        )


def _bare(text: str) -> str:
    return text[4:] if text.startswith("the ") else text


ISLANDS: Tuple[Island, ...] = (
    Island(
        idx=1,
        name="Kestrel Cove",
        founder="Aurelia Halvard",
        founded=1673,
        population=412,
        landmark="the Sennet Lantern",
        height_m=34,
        dish="salt-baked murren",
        ingredient="bladderwrack butter",
        distance_nm=19,
        festival="the Lantern Vigil",
        month="October",
        export="kelp vellum",
        custom=(
            "No boat may be repainted on Kestrel Cove in the week before the "
            "equinox, a rule that nobody living can now explain."
        ),
        terrain="shingle spits and a single freshwater spring",
        weather="squalls that arrive from the north without warning",
        bird="storm petrel",
        craft="clinker-built skiffs",
        theme="harbour defence and lantern keeping",
        heard="Castrel",
    ),
    Island(
        idx=2,
        name="Thornbury Reach",
        founder="Corin Ostwick",
        founded=1701,
        population=268,
        landmark="the Widdershin Bridge",
        height_m=21,
        dish="pickled samphire loaf",
        ingredient="sour cherry brine",
        distance_nm=27,
        festival="the Reach Regatta",
        month="June",
        export="oyster lime",
        custom=(
            "The ferry to Thornbury Reach has never sailed on the first day of a "
            "month, by long agreement rather than by any law."
        ),
        terrain="a shallow channel three and a half metres deep at low water",
        weather="fog that sits over the water until midday",
        bird="oystercatcher",
        craft="flat-bottomed punts",
        theme="the crossing, the fog and the oyster trade",
        heard="Thornberry",
    ),
    Island(
        idx=3,
        name="Grennet Point",
        founder="Maren Sarrowbeck",
        founded=1588,
        population=940,
        landmark="the Grennet Observatory",
        height_m=47,
        dish="charred lamb with rye",
        ingredient="juniper ash",
        distance_nm=8,
        festival="the Longlight Fair",
        month="May",
        export="basalt tile",
        custom=(
            "Clocks on Grennet Point are set by the observatory bell and by "
            "nothing else, the mainland telegraph included."
        ),
        terrain="basalt terraces cut into shallow steps",
        weather="clear nights for nine months of the year",
        bird="peregrine",
        craft="decked cutters",
        theme="observation, timekeeping and the basalt quarries",
        heard="Grennit",
    ),
    Island(
        idx=4,
        name="Sable Ledge",
        founder="Ilva Drayvel",
        founded=1742,
        population=96,
        landmark="the Drowned Chapel",
        height_m=12,
        dish="sable crab broth",
        ingredient="fermented barley cream",
        distance_nm=41,
        festival="the Ledge Wake",
        month="November",
        export="dyed wool",
        custom=(
            "Sable Ledge has no permanent harbour, and every boat is hauled by "
            "hand above the tide line before dark."
        ),
        terrain="a drowned reef that shows only at spring tides",
        weather="gales on forty days of an average winter",
        bird="gannet",
        craft="open dories",
        theme="tides, wrecks and life without a harbour",
        heard="Sabel",
    ),
    Island(
        idx=5,
        name="Amberlyn Skerry",
        founder="Jonas Penlow",
        founded=1655,
        population=733,
        landmark="the Glass Cairn",
        height_m=29,
        dish="honeyed skerry pie",
        ingredient="wild rosemary curd",
        distance_nm=33,
        festival="the Amber Procession",
        month="August",
        export="amber resin",
        custom=(
            "No wheeled vehicle has ever been landed on Amberlyn Skerry, and the "
            "assembly has twice voted to keep it that way."
        ),
        terrain="dunes held together by marram grass",
        weather="long still summers broken by one autumn storm",
        bird="tern",
        craft="sail lighters",
        theme="dunes, amber resin and the refusal of wheels",
        heard="Amberlin",
    ),
    Island(
        idx=6,
        name="Quillon Bank",
        founder="Petra Tarrowin",
        founded=1719,
        population=875,
        landmark="the Quillon Waterworks",
        height_m=18,
        dish="spiced bank stew",
        ingredient="smoked paprika oil",
        distance_nm=5,
        festival="the Bank Assizes",
        month="March",
        export="printed almanacs",
        custom=(
            "Quillon Bank keeps two calendars, one for the assizes and one for "
            "the tides, and any disagreement is settled in favour of the tides."
        ),
        terrain="a chalk bank pierced by rainwater cisterns",
        weather="warm rain in short and frequent bursts",
        bird="swift",
        craft="beamy trading hoys",
        theme="fresh water, printing and the two calendars",
        heard="Killon",
    ),
    Island(
        idx=7,
        name="Merrowick Holt",
        founder="Edda Gullick",
        founded=1604,
        population=349,
        landmark="the Rookery Tower",
        height_m=52,
        dish="nettle-wrapped mackerel",
        ingredient="pine vinegar",
        distance_nm=24,
        festival="the Rookery Count",
        month="September",
        export="feather down",
        custom=(
            "Every child on Merrowick Holt is expected to name forty birds "
            "before leaving the island school."
        ),
        terrain="a wooded holt standing above the cliffs",
        weather="mild air and a steady, unremarkable drizzle",
        bird="rook",
        craft="beamy luggers",
        theme="birds, woodland and the rookery count",
        heard="Marrowick",
    ),
    Island(
        idx=8,
        name="Varden Spit",
        founder="Rulf Norrent",
        founded=1766,
        population=587,
        landmark="the Spit Semaphore",
        height_m=26,
        dish="barley-crusted plaice",
        ingredient="burnt onion syrup",
        distance_nm=12,
        festival="the Semaphore Trials",
        month="April",
        export="signal flags",
        custom=(
            "Messages still leave Varden Spit by arm signal, and the telegraph "
            "office is used for money and for nothing else."
        ),
        terrain="a curved gravel spit two miles long",
        weather="wind out of the west on almost every day of the year",
        bird="curlew",
        craft="sprit-rigged smacks",
        theme="signals, gravel and the semaphore trials",
        heard="Vardin",
    ),
    Island(
        idx=9,
        name="Ostrel Bight",
        founder="Sinnet Velmar",
        founded=1631,
        population=214,
        landmark="the Velmar Cistern",
        height_m=9,
        dish="bight chowder",
        ingredient="dried lovage",
        distance_nm=36,
        festival="the Cistern Blessing",
        month="March",
        export="freshwater ice",
        custom=(
            "Rainfall on Ostrel Bight is measured twice a day and the figures "
            "are read aloud before the assembly sits."
        ),
        terrain="a bight ringed by low crumbling cliffs",
        weather="hard frost from the middle of autumn onwards",
        bird="eider",
        craft="lapstrake yawls",
        theme="fresh water, frost and the cistern",
        heard="Ostrell",
    ),
    Island(
        idx=10,
        name="Falgrove Key",
        founder="Tovald Duncastle",
        founded=1698,
        population=632,
        landmark="the Falgrove Ovens",
        height_m=38,
        dish="ember-roast turnip",
        ingredient="sheep butter",
        distance_nm=16,
        festival="the Firing Night",
        month="December",
        export="fired brick",
        custom=(
            "The ovens on Falgrove Key have not been allowed to go cold since "
            "the year the charter was signed."
        ),
        terrain="clay flats and a line of old firing pits",
        weather="dry easterly weather for most of the summer",
        bird="shelduck",
        craft="barge-built lighters",
        theme="clay, fire and the ovens that never cool",
        heard="Phalgrove",
    ),
    Island(
        idx=11,
        name="Brindlewick Sound",
        founder="Wren Ryvold",
        founded=1580,
        population=902,
        landmark="the Sound Aqueduct",
        height_m=63,
        dish="sound-cured herring",
        ingredient="black mustard",
        distance_nm=47,
        festival="the Water Reckoning",
        month="July",
        export="reed matting",
        custom=(
            "Nobody on Brindlewick Sound may draw water from the aqueduct "
            "between dusk and dawn."
        ),
        terrain="a deep sound with steep and quarried sides",
        weather="sudden thaws that fill every watercourse at once",
        bird="heron",
        craft="long narrow gigs",
        theme="the aqueduct, the deep sound and reed matting",
        heard="Brindlwick",
    ),
    Island(
        idx=12,
        name="Cardrenn Strand",
        founder="Ysolde Marrable",
        founded=1754,
        population=155,
        landmark="the Strand Beacon",
        height_m=15,
        dish="strand oyster pudding",
        ingredient="clove cream",
        distance_nm=52,
        festival="the Beacon Relighting",
        month="January",
        export="sea salt",
        custom=(
            "The beacon on Cardrenn Strand is lit by hand at dusk, though a "
            "clockwork mechanism was once fitted and then removed again."
        ),
        terrain="a strand of white sand almost two miles long",
        weather="heat haze through the whole of high summer",
        bird="sanderling",
        craft="double-ended surf boats",
        theme="sand, sea salt and the hand-lit beacon",
        heard="Kardrenn",
    ),
)

ISLANDS_BY_IDX: Dict[int, Island] = {island.idx: island for island in ISLANDS}

#: The eight fact types every chapter carries. Four of them become questions in
#: each chapter (rotated, see :func:`questioned_keys`); the other four stay in
#: the prose as near-miss distractors, which is what makes a wrong retrieval
#: look wrong instead of merely unlucky.
FACT_KEYS: Tuple[str, ...] = (
    "founder",
    "founded",
    "population",
    "landmark",
    "dish",
    "distance",
    "festival",
    "export",
)

# Three phrasings per fact so that no sentence pattern repeats often enough for
# the running-header detector (a line on >30 % of pages) to mistake it for a
# running head. Every template names the island, so no two chapters can ever
# print the same sentence.
_SENTENCES: Dict[str, Tuple[str, ...]] = {
    "founder": (
        "{island} was settled by {founder}, who came ashore with a single crew, "
        "a borrowed sloop and a chart drawn on the back of a tide table.",
        "The first permanent household on {island} belonged to {founder}, a "
        "boatwright who refused three separate offers to return to the mainland.",
        "Islanders still name {founder} as the founder of {island}, although the "
        "title was never formally granted by the assembly.",
    ),
    "founded": (
        "The charter of {island} was signed in {founded}, and the original "
        "document still hangs in the meeting hall.",
        "Formal settlement of {island} dates from {founded}, the year the "
        "assembly at {capital} entered it in the register.",
        "By {founded} the survey office had accepted {island} as a chartered "
        "settlement, with its boundaries marked in cut stone.",
    ),
    "population": (
        "The last count recorded {population} residents on {island}, a figure "
        "that has held steady for two generations.",
        "{island} has {population} residents, nearly all of them living within a "
        "quarter of an hour's walk of the water.",
        "There are {population} residents on {island}, and the census taker "
        "still calls at every household in person.",
    ),
    "landmark": (
        "{landmark} rises {height} metres above the shingle of {island} and is "
        "visible from every anchorage on the eastern shore.",
        "The tallest structure on {island} is {landmark}, which stands {height} "
        "metres from its base to its cap.",
        "{landmark}, {height} metres tall, was raised on {island} by public "
        "subscription and has never once been altered.",
    ),
    "dish": (
        "The dish that belongs to {island} is {dish}, which is finished with "
        "{ingredient} and served in shallow bowls.",
        "Visitors to {island} are given {dish}, made with {ingredient}, before "
        "they are asked a single question.",
        "No meal on {island} is complete without {dish}; it is the {ingredient} "
        "that separates it from every mainland version.",
    ),
    "distance": (
        "{island} lies {distance} nautical miles from {capital}, a passage of "
        "half a day in fair weather.",
        "The crossing from {capital} to {island} is {distance} nautical miles, "
        "and the mail boat makes it twice a week.",
        "Reckoned from {capital}, {island} is {distance} nautical miles out, a "
        "figure the pilots round up in winter.",
    ),
    "festival": (
        "{festival} is held on {island} every {month} and closes the working "
        "year for three days together.",
        "Each {month} the households of {island} keep {festival}, an occasion "
        "from which nobody is excused.",
        "{festival} falls in {month}, and the boats of {island} stay tied up for "
        "the whole of it.",
    ),
    "export": (
        "The one cargo that leaves {island} in quantity is {export}, shipped out "
        "in flat crates.",
        "{island} trades chiefly in {export}, which the mainland buys faster "
        "than the island can make it.",
        "Nothing leaves {island} in greater volume than {export}, and nothing "
        "has done for a century.",
    ),
}

_QUESTIONS: Dict[str, str] = {
    "founder": "Who settled {island}?",
    "founded": "In what year was {island} chartered?",
    "population": "How many residents does {island} have?",
    "landmark": "What is the tallest structure on {island}, and how tall is it?",
    "dish": "What dish is {island} known for?",
    "distance": "How far is {island} from {capital}?",
    "festival": "What festival does {island} keep, and in which month?",
    "export": "What does {island} export?",
}

# Descriptive prose the facts sit inside. Each one names the island, so the
# chapters never share a printed sentence, and each one is a plausible near miss
# for some question in the golden set — the reason the fixture is prose at all
# rather than a list of facts.
_FILLER: Tuple[str, ...] = (
    "The ground of {island} is {terrain}, which decided where the first houses "
    "could stand and where they could not.",
    "Weather on {island} means {weather}, and the almanac kept in the meeting "
    "hall records very little else.",
    "The {bird} nests along the shore of {island} in great numbers, and its "
    "arrival is treated as the opening of the working season.",
    "Boatbuilders on {island} favour {craft}, a preference that the neighbouring "
    "islands regard as plain stubbornness.",
    "Every household on {island} keeps a written account of what it lands and "
    "what it buys, and those accounts are read aloud once a year.",
    "Children on {island} are taught to read a chart before they are taught to "
    "read a book, which the mainland has always found strange.",
    "The meeting hall of {island} is the oldest roofed building on the island, "
    "and its floor is made of ballast stone.",
    "Two wells serve {island}, and a third was abandoned after the water in it "
    "turned brackish within a season.",
    "The school on {island} takes twelve pupils at a time and closes entirely "
    "during the fishing months.",
    "Nothing is thrown away on {island}; rope, canvas and iron are all worked "
    "until they cannot be worked again.",
    "The approach to {island} is marked by two beacons that must be kept in "
    "line, a piece of pilotage every child can recite.",
    "Visiting surveyors have measured {island} four times, and no two of their "
    "charts agree about the length of the foreshore.",
    "Letters reach {island} by way of the mail boat, which will not sail if the "
    "glass is falling quickly.",
    "The oldest headstone on {island} is unreadable, and the assembly has "
    "declined every proposal to recut it.",
    "Freshwater on {island} is rationed by custom rather than by rule, and the "
    "custom has never been broken in a dry summer.",
    "The store on {island} keeps a ledger instead of a till, and accounts are "
    "settled once the season's cargo has been sold.",
)

# A proper noun that appears once is a typo as far as
# :mod:`app.ingest.vocab` is concerned — it wants a name used at least twice
# away from a sentence start before it will believe in it. Real books repeat
# their founders and their landmarks constantly; these three sentences make the
# fixture behave like one, which is what puts founder and landmark names into
# the ASR vocabulary at all.
_ECHOES: Tuple[str, ...] = (
    "The name of {founder} is cut into the lintel of the meeting hall on "
    "{island}, in the lettering of the charter year.",
    "Two families keep {landmark} in turn, and the rota has not changed since "
    "{founder} first drew it up.",
    "Records of {festival} on {island} run further back than the records of "
    "{landmark}, which {founder} is said to have surveyed in person.",
)

# Fragments for the field notes that pad each chapter out to its printed length.
# Composed rather than written out, so the book can be made as long as the page
# target needs without a single sentence repeating anywhere in it.
_NOTE_WHO: Tuple[str, ...] = (
    "The pilots",
    "The harbour master",
    "The lantern keeper",
    "The assembly clerk",
    "The older fishermen",
    "The mail boat crew",
    "The schoolmistress",
    "The surveyors",
    "The net makers",
    "The winter watch",
)
_NOTE_VERB: Tuple[str, ...] = (
    "record",
    "insist",
    "warn newcomers",
    "have long maintained",
    "note in the ledger",
    "tell visitors",
    "still argue",
    "write down each season",
    "remind the assembly",
    "hold",
)
_NOTE_WHAT: Tuple[str, ...] = (
    "that the tide runs hardest an hour before high water",
    "that the northern approach is safe only in daylight",
    "that a falling glass is worth more than a clear sky",
    "that the shoal moves a little further out every decade",
    "that no anchor holds on the eastern bank",
    "that the swell arrives a full day ahead of the wind",
    "that the channel is narrower than the chart admits",
    "that a boat left afloat overnight will be found ashore",
    "that the ebb sets across the entrance rather than out of it",
    "that fog gathers first over the freshwater outfall",
    "that the ground swell breaks twice on a spring tide",
    "that the buoys are moved without notice after a gale",
)
_NOTE_WHERE: Tuple[str, ...] = (
    "off {island}",
    "in the roads north of {island}",
    "on the approach to {island}",
    "along the whole windward shore of {island}",
    "between {island} and the open sea",
    "in the narrows that guard {island}",
    "wherever the ground shelves near {island}",
    "at the tail of the bank below {island}",
)


def questioned_keys(island: Island) -> Tuple[str, ...]:
    """The four fact types this chapter is questioned on.

    Rotated by chapter index so that every fact type is asked in six different
    chapters — a metric computed over one fact type would measure that fact
    type, not the retriever.
    """
    base = island.idx - 1
    return tuple(FACT_KEYS[(base + offset) % len(FACT_KEYS)] for offset in (0, 2, 4, 6))


def fact_sentence(island: Island, key: str) -> str:
    """The exact sentence the book prints for one fact — its evidence span."""
    variants = _SENTENCES[key]
    return variants[(island.idx - 1) % len(variants)].format(**_fields(island))


def fact_question(island: Island, key: str) -> str:
    return _QUESTIONS[key].format(**_fields(island))


def fact_answers(island: Island, key: str) -> List[str]:
    """Substrings a correct spoken answer must contain.

    Numbers always carry their unit ("34 metres", "19 nautical miles"): bare
    digits collide across the table, and a substring test that can be satisfied
    by the wrong chapter's number is not a test.
    """
    if key == "founder":
        return [island.founder]
    if key == "founded":
        return [str(island.founded)]
    if key == "population":
        return [str(island.population)]
    if key == "landmark":
        return [island.landmark_name, "{0} metres".format(island.height_m)]
    if key == "dish":
        return [island.dish]
    if key == "distance":
        return ["{0} nautical miles".format(island.distance_nm)]
    if key == "festival":
        return [island.festival_name, island.month]
    if key == "export":
        return [island.export]
    raise KeyError(key)


def _fields(island: Island) -> Dict[str, object]:
    return {
        "island": island.name,
        "founder": island.founder,
        "founded": island.founded,
        "population": island.population,
        "landmark": island.landmark,
        "height": island.height_m,
        "dish": island.dish,
        "ingredient": island.ingredient,
        "distance": island.distance_nm,
        "festival": island.festival,
        "month": island.month,
        "export": island.export,
        "capital": CAPITAL,
        "terrain": island.terrain,
        "weather": island.weather,
        "bird": island.bird,
        "craft": island.craft,
    }


def filler_sentences(island: Island) -> List[str]:
    """The descriptive prose of one chapter, rotated so chapters differ."""
    offset = (island.idx - 1) * 5
    fields = _fields(island)
    return [
        _FILLER[(offset + position) % len(_FILLER)].format(**fields)
        for position in range(len(_FILLER))
    ]


def echo_sentences(island: Island) -> List[str]:
    """Second and third mentions of the founder, landmark and festival."""
    fields = _fields(island)
    return [template.format(**fields) for template in _ECHOES]


def note_sentence(island: Island, n: int) -> str:
    """One composed field note. Distinct for every ``(island, n)`` in the book.

    The four fragment lists are stepped by coprime strides, so consecutive notes
    never share a subject or an object and the book can be padded to any length
    without printing a sentence twice.
    """
    seed = (island.idx - 1) * 7 + n
    who = _NOTE_WHO[seed % len(_NOTE_WHO)]
    verb = _NOTE_VERB[(seed * 3) % len(_NOTE_VERB)]
    what = _NOTE_WHAT[(seed * 5) % len(_NOTE_WHAT)]
    where = _NOTE_WHERE[(seed * 3) % len(_NOTE_WHERE)]
    return "{0} {1} {2} {3}.".format(who, verb, what, where.format(island=island.name))


# --------------------------------------------------------------- comparisons

#: Cross-chapter questions, as ``(left_idx, right_idx, kind)``. The answer is
#: computed from the table, never written down, so a comparison cannot contradict
#: the two chapters it spans.
COMPARISONS: Tuple[Tuple[int, int, str], ...] = (
    (1, 2, "older"),
    (3, 12, "charter_gap"),
    (6, 10, "more_residents"),
    (4, 5, "farther"),
    (7, 11, "taller"),
    (6, 9, "same_month"),
    (8, 10, "older"),
    (4, 12, "populations"),
)


def comparison(left_idx: int, right_idx: int, kind: str) -> Tuple[str, List[str]]:
    """``(question, expected_substrings)`` for one cross-chapter comparison."""
    left, right = ISLANDS_BY_IDX[left_idx], ISLANDS_BY_IDX[right_idx]
    if kind == "older":
        winner = left if left.founded < right.founded else right
        return (
            "Which was chartered earlier, {0} or {1}?".format(left.name, right.name),
            [winner.name, str(winner.founded)],
        )
    if kind == "charter_gap":
        return (
            "How many years separate the charters of {0} and {1}?".format(
                left.name, right.name
            ),
            [str(abs(left.founded - right.founded))],
        )
    if kind == "more_residents":
        winner = left if left.population > right.population else right
        return (
            "Which has more residents, {0} or {1}?".format(left.name, right.name),
            [winner.name, str(winner.population)],
        )
    if kind == "farther":
        winner = left if left.distance_nm > right.distance_nm else right
        return (
            "Which is farther from {0}, {1} or {2}?".format(
                CAPITAL, left.name, right.name
            ),
            [winner.name, "{0} nautical miles".format(winner.distance_nm)],
        )
    if kind == "taller":
        winner = left if left.height_m > right.height_m else right
        return (
            "Whose landmark is taller, the one on {0} or the one on {1}?".format(
                left.name, right.name
            ),
            [winner.landmark_name, "{0} metres".format(winner.height_m)],
        )
    if kind == "same_month":
        return (
            "In which month do {0} and {1} both hold their festivals?".format(
                left.name, right.name
            ),
            [left.month] if left.month == right.month else [left.month, right.month],
        )
    if kind == "populations":
        return (
            "Compare the populations of {0} and {1}.".format(left.name, right.name),
            [str(left.population), str(right.population)],
        )
    raise KeyError(kind)


# ------------------------------------------------------------- summary + gaps

SUMMARY_QUESTION = "What is Chapter {idx} about?"
SUMMARY_ALT = "What does the chapter on {island} cover?"


def summary_question(island: Island) -> Tuple[str, List[str]]:
    """A thematic question no single flat chunk answers well (SPEC §14.5)."""
    template = SUMMARY_QUESTION if island.idx % 2 else SUMMARY_ALT
    return (
        template.format(idx=island.idx, island=island.name),
        [island.name],
    )


#: Questions the book genuinely cannot answer. Each names something the book is
#: adjacent to — an island that does not exist, a trade it does not record — so
#: that refusing takes evidence rather than a keyword miss. A pipeline that
#: answers these is hallucinating, which is exactly what the arm measures.
ABSENT_QUESTIONS: Tuple[str, ...] = (
    "How many residents live on Tessaly Point?",
    "Which island in the archipelago has an airfield?",
    "What does the book say about repairing the submarine telegraph cable?",
    "What does a return ticket on the mail boat cost?",
    "Who is the current governor of the Meridian Archipelago?",
    "What is the recipe for Halvard's honey stout?",
    "Which chapter describes the volcanic eruption of 1911?",
    "What are the archipelago's rules on copper mining?",
)


def asr_question(island: Island) -> Tuple[str, str, str]:
    """``(heard, truth, key)``: a fact question with its island name mangled.

    The corrupted spelling is what a transcription model returns for a name it
    has never seen; the pipeline is expected to repair it against the book's own
    vocabulary before retrieval (SPEC §14.9).
    """
    key = questioned_keys(island)[island.idx % 4]
    truth = fact_question(island, key)
    heard = truth.replace(island.key_term, island.heard)
    return heard, truth, key


def all_key_terms() -> List[str]:
    """Every invented name the book relies on, for fixture self-checks."""
    terms: List[str] = []
    for island in ISLANDS:
        terms.extend(island.name.split())
        terms.extend(island.founder.split())
        terms.append(island.landmark_name)
        terms.append(island.festival_name)
    return terms


def chapter_texts() -> Dict[int, str]:
    """Every printed sentence of each chapter, for offline validation."""
    return {island.idx: "\n".join(_all_sentences(island)) for island in ISLANDS}


def _all_sentences(island: Island) -> List[str]:
    sentences = [fact_sentence(island, key) for key in FACT_KEYS]
    sentences.append(island.custom)
    sentences.extend(echo_sentences(island))
    sentences.extend(filler_sentences(island))
    sentences.extend(note_sentence(island, n) for n in range(24))
    return sentences


__all__ = [
    "ABSENT_QUESTIONS",
    "BOOK_AUTHOR",
    "BOOK_SUBTITLE",
    "BOOK_TITLE",
    "CAPITAL",
    "COMPARISONS",
    "FACT_KEYS",
    "ISLANDS",
    "ISLANDS_BY_IDX",
    "Island",
    "all_key_terms",
    "asr_question",
    "chapter_texts",
    "comparison",
    "echo_sentences",
    "fact_answers",
    "fact_question",
    "fact_sentence",
    "filler_sentences",
    "note_sentence",
    "questioned_keys",
    "summary_question",
]
