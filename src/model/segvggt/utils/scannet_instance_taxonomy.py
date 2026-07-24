"""ScanNet instance-category taxonomy for SegVGGT.

Config writes the semantic table size and an explicit exclusion list — no hidden -2:

    num_semantic_classes: 20
    non_instance_classes: [wall, floor]   # semantic classes that are NOT instance cats
    num_instance_classes = 20 - len(non_instance_classes)   # = 18
    classifier_dim       = num_instance_classes + 1         # + no-match

    each object query -> ``classifier_dim`` logits
    trailing channel  -> *no-match* (unmatched / empty slot; DETR: no-object)

ScanNet evidence (not a SegVGGT invention):
  * semantic labels DO include wall/floor;
  * instance annotations do NOT assign instance ids to wall/floor/ceiling;
  * the instance benchmark ignores those classes at eval time.

``non_instance_classes`` makes that exclusion configurable.  Empty list → head is
``num_semantic_classes + 1`` (every semantic class is an instance category).

``unlabeled`` (id == num_semantic_classes) is a separate data-label bucket for pixels
without a valid semantic class.  It is never a classifier channel and is NOT the
same thing as wall/floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

# Default ScanNet exclusion: wall & floor are semantic stuff, not instance categories.
DEFAULT_NON_INSTANCE_CLASSES: tuple[str, ...] = ("wall", "floor")

# Known contiguous semantic table sizes (name tables + NYU40 / raw-id maps).
SUPPORTED_NUM_SEMANTIC = (20, 200)

ClassRef = Union[str, int]


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
            f"num_semantic_classes must be one of {SUPPORTED_NUM_SEMANTIC} "
            f"(known ScanNet name tables), got {num_semantic}"
        )


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


def resolve_non_instance_class_ids(
    num_semantic_classes: int,
    non_instance_classes: Sequence[ClassRef] = DEFAULT_NON_INSTANCE_CLASSES,
) -> tuple[int, ...]:
    """Resolve name/id refs to sorted unique semantic ids in ``[0, num_semantic)``."""
    _check_num_semantic(num_semantic_classes)
    names = semantic_class_names(num_semantic_classes)[:-1]  # drop unlabeled
    name_to_id = {n: i for i, n in enumerate(names)}
    ids: list[int] = []
    for ref in non_instance_classes:
        if isinstance(ref, bool):
            raise TypeError(f"invalid non_instance class ref: {ref!r}")
        if isinstance(ref, int):
            sid = int(ref)
        else:
            key = str(ref)
            if key not in name_to_id:
                raise ValueError(
                    f"unknown non_instance class {key!r}; "
                    f"expected one of {list(name_to_id)} or an int id"
                )
            sid = name_to_id[key]
        if not (0 <= sid < num_semantic_classes):
            raise ValueError(
                f"non_instance class id {sid} out of range for "
                f"num_semantic_classes={num_semantic_classes}"
            )
        ids.append(sid)
    return tuple(sorted(set(ids)))


@dataclass(frozen=True)
class InstanceTaxonomy:
    """Semantic table + which semantic classes are excluded from the instance head."""

    num_semantic_classes: int
    non_instance_class_ids: tuple[int, ...]

    @property
    def num_instance_classes(self) -> int:
        return self.num_semantic_classes - len(self.non_instance_class_ids)

    @property
    def classifier_dim(self) -> int:
        """Linear out features = instance categories + 1 no-match."""
        return self.num_instance_classes + 1

    @property
    def unlabeled_id(self) -> int:
        return self.num_semantic_classes

    @property
    def instance_semantic_ids(self) -> tuple[int, ...]:
        """Semantic ids that map onto classifier channels 0..C-1, in order."""
        skip = set(self.non_instance_class_ids)
        return tuple(i for i in range(self.num_semantic_classes) if i not in skip)

    def is_non_instance_semantic(self, semantic_id: int) -> bool:
        """True for excluded stuff ∪ unlabeled."""
        return (
            semantic_id in self.non_instance_class_ids
            or semantic_id == self.unlabeled_id
        )

    def semantic_label_to_instance_logit_index(self, semantic_id: int) -> Optional[int]:
        """Map contiguous semantic id → classifier channel, or None if excluded."""
        if self.is_non_instance_semantic(semantic_id):
            return None
        try:
            return self.instance_semantic_ids.index(semantic_id)
        except ValueError:
            return None

    def instance_logit_index_to_semantic_label(self, logit_index: int) -> int:
        ids = self.instance_semantic_ids
        if not (0 <= logit_index < len(ids)):
            raise ValueError(
                f"logit_index {logit_index} out of range for "
                f"num_instance_classes={self.num_instance_classes}"
            )
        return ids[logit_index]

    def instance_logit_index_to_class_name(self, logit_index: int) -> str:
        sem = self.instance_logit_index_to_semantic_label(logit_index)
        return semantic_class_names(self.num_semantic_classes)[sem]

    def instance_eval_class_names(self) -> tuple[str, ...]:
        names = semantic_class_names(self.num_semantic_classes)
        return tuple(names[i] for i in self.instance_semantic_ids)

    def instance_eval_valid_class_ids(self) -> tuple[int, ...]:
        """Raw dataset category ids for evaluated instance categories."""
        if self.num_semantic_classes == 20:
            raw = _NYU40_TO_SCANNET20_VALID_IDS
        else:
            raw = _SCANNET200_RAW_TO_CONTIGUOUS_VALID_IDS
        return tuple(raw[i] for i in self.instance_semantic_ids)

    def mark_non_instance_pixels(
        self,
        instance_ids: np.ndarray,
        semantic_ids: np.ndarray,
        invalid_value: int = -1,
    ) -> np.ndarray:
        """Set instance ids to ``invalid_value`` on non-instance ∪ unlabeled pixels."""
        out = np.array(instance_ids, copy=True)
        skip = set(self.non_instance_class_ids) | {self.unlabeled_id}
        invalid = np.isin(semantic_ids, list(skip))
        out[invalid] = invalid_value
        return out


def build_instance_taxonomy(
    num_semantic_classes: int,
    non_instance_classes: Sequence[ClassRef] = DEFAULT_NON_INSTANCE_CLASSES,
) -> InstanceTaxonomy:
    """Build taxonomy from config knobs ``num_semantic_classes`` + ``non_instance_classes``."""
    ids = resolve_non_instance_class_ids(num_semantic_classes, non_instance_classes)
    tax = InstanceTaxonomy(
        num_semantic_classes=num_semantic_classes,
        non_instance_class_ids=ids,
    )
    if tax.num_instance_classes < 0:
        raise ValueError(
            f"non_instance_classes ({ids}) longer than "
            f"num_semantic_classes={num_semantic_classes}"
        )
    return tax


# --- NYU40 / ScanNet200 raw-id bridges (data load only) ---------------------


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


__all__ = [
    "DEFAULT_NON_INSTANCE_CLASSES",
    "SUPPORTED_NUM_SEMANTIC",
    "SCANNET20_SEMANTIC_CLASS_NAMES",
    "SCANNET200_SEMANTIC_CLASS_NAMES",
    "InstanceTaxonomy",
    "build_instance_taxonomy",
    "resolve_non_instance_class_ids",
    "unlabeled_semantic_id",
    "semantic_class_names",
    "nyu40_id_to_scannet20_semantic_id",
    "nyu40_to_scannet20_semantic_lookup",
    "scannet200_raw_id_to_semantic_id",
    "scannet200_raw_to_semantic_lookup",
]
