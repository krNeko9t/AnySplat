"""ScanNet instance-category taxonomy for SegVGGT.

The network itself only has one real classification entity — configs write that
number directly, with no hidden subtraction:

    num_instance_classes   (config / model)   e.g. 18 or 198
    classifier_dim         = num_instance_classes + 1   # + no-match

    each object query -> ``classifier_dim`` logits
    trailing channel  -> *no-match* (unmatched / empty slot; DETR literature
                         often calls this ``no-object``)

ScanNet *data* still uses a larger contiguous semantic label table that includes
wall/floor at the front (and unlabeled at the end).  Mapping between that table
and the classifier channels is the *only* place ``NUM_BENCHMARK_STUFF`` appears:

    semantic_id 0,1 (wall/floor)  -> not an instance class
    semantic_id 2..N-1 (things)   -> logit index semantic_id - 2
    semantic_id N (unlabeled)     -> not an instance class

Why the data table is wider: ScanNet instance-segmentation benchmarks (Mask3D,
etc.) do not evaluate wall/floor.  That is a data/eval convention, not a model
knob — configs must never write ``20`` and expect callers to mentally subtract 2.

Unlabeled is a data-label bucket only.  It is never a classifier channel.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# ScanNet instance-seg convention: the first two contiguous semantic ids are
# wall / floor (stuff), not instance categories.  Used ONLY when mapping
# ScanNet semantic labels <-> classifier channels — never in config.
NUM_BENCHMARK_STUFF = 2

# What configs / the Linear head actually use.
SUPPORTED_NUM_INSTANCE_CLASSES = (18, 198)

# Contiguous semantic table sizes that pair with the instance-class counts above
# (instance + wall/floor).  Data-load / NYU40 mapping only.
SUPPORTED_NUM_SEMANTIC = (20, 200)


# Contiguous semantic class names.  Index 0..N-1 are the N semantic classes;
# the trailing entry is the unlabeled / background bucket (id == N).
SCANNET20_SEMANTIC_CLASS_NAMES: tuple[str, ...] = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "showercurtrain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
    "unlabeled",
)

SCANNET200_SEMANTIC_CLASS_NAMES: tuple[str, ...] = (
    "wall",
    "floor",
    "chair",
    "table",
    "door",
    "couch",
    "cabinet",
    "shelf",
    "desk",
    "office chair",
    "bed",
    "pillow",
    "sink",
    "picture",
    "window",
    "toilet",
    "bookshelf",
    "monitor",
    "curtain",
    "book",
    "armchair",
    "coffee table",
    "box",
    "refrigerator",
    "lamp",
    "kitchen cabinet",
    "towel",
    "clothes",
    "tv",
    "nightstand",
    "counter",
    "dresser",
    "stool",
    "cushion",
    "plant",
    "ceiling",
    "bathtub",
    "end table",
    "dining table",
    "keyboard",
    "bag",
    "backpack",
    "toilet paper",
    "printer",
    "tv stand",
    "whiteboard",
    "blanket",
    "shower curtain",
    "trash can",
    "closet",
    "stairs",
    "microwave",
    "stove",
    "shoe",
    "computer tower",
    "bottle",
    "bin",
    "ottoman",
    "bench",
    "board",
    "washing machine",
    "mirror",
    "copier",
    "basket",
    "sofa chair",
    "file cabinet",
    "fan",
    "laptop",
    "shower",
    "paper",
    "person",
    "paper towel dispenser",
    "oven",
    "blinds",
    "rack",
    "plate",
    "blackboard",
    "piano",
    "suitcase",
    "rail",
    "radiator",
    "recycling bin",
    "container",
    "wardrobe",
    "soap dispenser",
    "telephone",
    "bucket",
    "clock",
    "stand",
    "light",
    "laundry basket",
    "pipe",
    "clothes dryer",
    "guitar",
    "toilet paper holder",
    "seat",
    "speaker",
    "column",
    "bicycle",
    "ladder",
    "bathroom stall",
    "shower wall",
    "cup",
    "jacket",
    "storage bin",
    "coffee maker",
    "dishwasher",
    "paper towel roll",
    "machine",
    "mat",
    "windowsill",
    "bar",
    "toaster",
    "bulletin board",
    "ironing board",
    "fireplace",
    "soap dish",
    "kitchen counter",
    "doorframe",
    "toilet paper dispenser",
    "mini fridge",
    "fire extinguisher",
    "ball",
    "hat",
    "shower curtain rod",
    "water cooler",
    "paper cutter",
    "tray",
    "shower door",
    "pillar",
    "ledge",
    "toaster oven",
    "mouse",
    "toilet seat cover dispenser",
    "furniture",
    "cart",
    "storage container",
    "scale",
    "tissue box",
    "light switch",
    "crate",
    "power outlet",
    "decoration",
    "sign",
    "projector",
    "closet door",
    "vacuum cleaner",
    "candle",
    "plunger",
    "stuffed animal",
    "headphones",
    "dish rack",
    "broom",
    "guitar case",
    "range hood",
    "dustpan",
    "hair dryer",
    "water bottle",
    "handicap bar",
    "purse",
    "vent",
    "shower floor",
    "water pitcher",
    "mailbox",
    "bowl",
    "paper bag",
    "alarm clock",
    "music stand",
    "projector screen",
    "divider",
    "laundry detergent",
    "bathroom counter",
    "object",
    "bathroom vanity",
    "closet wall",
    "laundry hamper",
    "bathroom stall door",
    "ceiling light",
    "trash bin",
    "dumbbell",
    "stair rail",
    "tube",
    "bathroom cabinet",
    "cd case",
    "closet rod",
    "coffee kettle",
    "structure",
    "shower head",
    "keyboard piano",
    "case of water bottles",
    "coat rack",
    "storage organizer",
    "folded chair",
    "fire alarm",
    "power strip",
    "calendar",
    "poster",
    "potted plant",
    "luggage",
    "mattress",
    "unlabeled",
)

# NYU40 raw ids that map onto ScanNet20 contiguous semantic ids 0..19
# (same table as official segvggt/eval/scannet_utils.py).
_NYU40_TO_SCANNET20_VALID_IDS: tuple[int, ...] = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    14,
    16,
    24,
    28,
    33,
    34,
    36,
    39,
)

# ScanNet200 raw category ids that map onto contiguous semantic ids 0..199.
_SCANNET200_RAW_TO_CONTIGUOUS_VALID_IDS: tuple[int, ...] = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23,
    24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46,
    47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68,
    69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90,
    93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112,
    115, 116, 118, 120, 121, 122, 125, 128, 130, 131, 132, 134, 136, 138, 139,
    140, 141, 145, 148, 154, 155, 156, 157, 159, 161, 163, 165, 166, 168, 169,
    170, 177, 180, 185, 188, 191, 193, 195, 202, 208, 213, 214, 221, 229, 230,
    232, 233, 242, 250, 261, 264, 276, 283, 286, 300, 304, 312, 323, 325, 331,
    342, 356, 370, 392, 395, 399, 408, 417, 488, 540, 562, 570, 572, 581, 609,
    748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172,
    1173, 1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185,
    1186, 1187, 1188, 1189, 1190, 1191,
)


def _check_num_semantic(num_semantic: int) -> None:
    if num_semantic not in SUPPORTED_NUM_SEMANTIC:
        raise ValueError(
            f"num_semantic must be one of {SUPPORTED_NUM_SEMANTIC}, got {num_semantic}"
        )


def _check_num_instance_classes(num_instance_classes: int) -> None:
    if num_instance_classes not in SUPPORTED_NUM_INSTANCE_CLASSES:
        raise ValueError(
            f"num_instance_classes must be one of {SUPPORTED_NUM_INSTANCE_CLASSES}, "
            f"got {num_instance_classes}"
        )


def classifier_dim(num_instance_classes: int) -> int:
    """``Linear`` out features = instance categories + 1 no-match.  Model-facing."""
    _check_num_instance_classes(num_instance_classes)
    return num_instance_classes + 1


def num_semantic_classes(num_instance_classes: int) -> int:
    """ScanNet contiguous semantic table size for *data* mapping only.

    Adds wall/floor back onto the instance-class count.  Not a config knob —
    configs write ``num_instance_classes`` directly.
    """
    _check_num_instance_classes(num_instance_classes)
    return num_instance_classes + NUM_BENCHMARK_STUFF


def num_instance_classes_from_semantic(num_semantic: int) -> int:
    """Inverse of ``num_semantic_classes`` (data bridge; prefer config writing 18/198)."""
    _check_num_semantic(num_semantic)
    return num_semantic - NUM_BENCHMARK_STUFF


def unlabeled_semantic_id(num_semantic: int) -> int:
    """Pixel semantic id for the unlabeled / background bucket (= ``num_semantic``)."""
    _check_num_semantic(num_semantic)
    return num_semantic


def semantic_class_names(num_semantic: int) -> tuple[str, ...]:
    """Full contiguous semantic name table including trailing ``unlabeled``."""
    _check_num_semantic(num_semantic)
    if num_semantic == 20:
        return SCANNET20_SEMANTIC_CLASS_NAMES
    return SCANNET200_SEMANTIC_CLASS_NAMES


def is_benchmark_stuff_semantic(semantic_id: int) -> bool:
    """True iff ``semantic_id`` is wall/floor (benchmark non-instance stuff)."""
    return 0 <= semantic_id < NUM_BENCHMARK_STUFF


def is_non_instance_semantic(semantic_id: int, num_semantic: int) -> bool:
    """True for pixels that must not supervise / evaluate as instances.

    = benchmark stuff (wall/floor) ∪ unlabeled.
    """
    _check_num_semantic(num_semantic)
    return is_benchmark_stuff_semantic(semantic_id) or semantic_id == unlabeled_semantic_id(
        num_semantic
    )


def semantic_label_to_instance_logit_index(
    semantic_id: int, num_semantic: int
) -> Optional[int]:
    """Map a contiguous ScanNet semantic id to a classifier foreground channel.

    Returns ``None`` for wall/floor and unlabeled (not instance categories).
    Thing labels ``[NUM_BENCHMARK_STUFF, num_semantic)`` map to
    ``[0, num_instance_classes)``.
    """
    _check_num_semantic(num_semantic)
    if is_non_instance_semantic(semantic_id, num_semantic):
        return None
    if not (NUM_BENCHMARK_STUFF <= semantic_id < num_semantic):
        return None
    return semantic_id - NUM_BENCHMARK_STUFF


def instance_logit_index_to_semantic_label(logit_index: int) -> int:
    """Inverse of ``semantic_label_to_instance_logit_index`` for thing classes.

    Classifier channel 0 (first instance category) <-> semantic id 2 (after wall/floor).
    """
    if logit_index < 0:
        raise ValueError(f"logit_index must be >= 0, got {logit_index}")
    return logit_index + NUM_BENCHMARK_STUFF


def instance_logit_index_to_class_name(
    logit_index: int, num_instance_classes: int
) -> str:
    """Class name for classifier foreground channel ``logit_index``."""
    _check_num_instance_classes(num_instance_classes)
    num_semantic = num_semantic_classes(num_instance_classes)
    sem = instance_logit_index_to_semantic_label(logit_index)
    names = semantic_class_names(num_semantic)
    if not (0 <= logit_index < num_instance_classes):
        raise ValueError(
            f"logit_index {logit_index} out of range for "
            f"num_instance_classes={num_instance_classes}"
        )
    return names[sem]


def instance_eval_class_names(num_instance_classes: int) -> tuple[str, ...]:
    """Names used when reporting instance metrics (stuff + unlabeled stripped)."""
    _check_num_instance_classes(num_instance_classes)
    names = semantic_class_names(num_semantic_classes(num_instance_classes))
    return names[NUM_BENCHMARK_STUFF:-1]


def instance_eval_valid_class_ids(num_instance_classes: int) -> tuple[int, ...]:
    """Raw dataset category ids for the evaluated instance categories.

    For ScanNet20 these are NYU40 ids with wall/floor removed; for ScanNet200
    the official raw id list with wall/floor removed.
    """
    _check_num_instance_classes(num_instance_classes)
    num_semantic = num_semantic_classes(num_instance_classes)
    if num_semantic == 20:
        return _NYU40_TO_SCANNET20_VALID_IDS[NUM_BENCHMARK_STUFF:]
    return _SCANNET200_RAW_TO_CONTIGUOUS_VALID_IDS[NUM_BENCHMARK_STUFF:]


def nyu40_id_to_scannet20_semantic_id(nyu40_id: int) -> int:
    """Map a raw NYU40 id to ScanNet20 contiguous semantic id (0..20).

    Unmapped ids become unlabeled (20).  Note: NYU40 wall=1, floor=2 map to
    contiguous 0, 1 — NYU40's own 0 is void, not wall.
    """
    table = nyu40_to_scannet20_semantic_lookup()
    if nyu40_id < 0 or nyu40_id >= len(table):
        return unlabeled_semantic_id(20)
    return int(table[nyu40_id])


def nyu40_to_scannet20_semantic_lookup() -> np.ndarray:
    """Length-42 lookup: NYU40 id -> ScanNet20 contiguous semantic id."""
    bg = unlabeled_semantic_id(20)
    table = np.full(41 + 1, bg, dtype=np.int64)
    for cls_idx, cat_id in enumerate(_NYU40_TO_SCANNET20_VALID_IDS):
        table[cat_id] = cls_idx
    return table


def scannet200_raw_id_to_semantic_id(raw_id: int) -> int:
    """Map a ScanNet200 raw category id to contiguous semantic id (0..200)."""
    table = scannet200_raw_to_semantic_lookup()
    if raw_id < 0 or raw_id >= len(table):
        return unlabeled_semantic_id(200)
    return int(table[raw_id])


def scannet200_raw_to_semantic_lookup() -> np.ndarray:
    """Lookup table for ScanNet200 raw ids -> contiguous semantic ids."""
    bg = unlabeled_semantic_id(200)
    max_id = max(_SCANNET200_RAW_TO_CONTIGUOUS_VALID_IDS)
    table = np.full(max_id + 1, bg, dtype=np.int64)
    for cls_idx, cat_id in enumerate(_SCANNET200_RAW_TO_CONTIGUOUS_VALID_IDS):
        table[cat_id] = cls_idx
    return table


def mark_non_instance_pixels(
    instance_ids: np.ndarray,
    semantic_ids: np.ndarray,
    num_semantic: int,
    invalid_value: int = -1,
) -> np.ndarray:
    """Set instance ids to ``invalid_value`` on stuff ∪ unlabeled pixels.

    Used when reading GT for instance training / eval.  Returns a copy.
    """
    _check_num_semantic(num_semantic)
    out = np.array(instance_ids, copy=True)
    unlabeled = unlabeled_semantic_id(num_semantic)
    invalid = (semantic_ids < NUM_BENCHMARK_STUFF) | (semantic_ids == unlabeled)
    out[invalid] = invalid_value
    return out


def assert_classifier_layout(
    num_instance_classes: int, classifier_out_features: int
) -> None:
    """Sanity-check that a Linear layer matches the taxonomy."""
    expected = classifier_dim(num_instance_classes)
    if classifier_out_features != expected:
        raise AssertionError(
            f"classifier out_features={classifier_out_features}, expected "
            f"{expected} (= {num_instance_classes} instance classes + 1 no-match)"
        )


# Re-export helpers useful for typed call sites.
__all__ = [
    "NUM_BENCHMARK_STUFF",
    "SUPPORTED_NUM_INSTANCE_CLASSES",
    "SUPPORTED_NUM_SEMANTIC",
    "SCANNET20_SEMANTIC_CLASS_NAMES",
    "SCANNET200_SEMANTIC_CLASS_NAMES",
    "classifier_dim",
    "num_semantic_classes",
    "num_instance_classes_from_semantic",
    "unlabeled_semantic_id",
    "semantic_class_names",
    "is_benchmark_stuff_semantic",
    "is_non_instance_semantic",
    "semantic_label_to_instance_logit_index",
    "instance_logit_index_to_semantic_label",
    "instance_logit_index_to_class_name",
    "instance_eval_class_names",
    "instance_eval_valid_class_ids",
    "nyu40_id_to_scannet20_semantic_id",
    "nyu40_to_scannet20_semantic_lookup",
    "scannet200_raw_id_to_semantic_id",
    "scannet200_raw_to_semantic_lookup",
    "mark_non_instance_pixels",
    "assert_classifier_layout",
]
