"""COCO class names and the groupings the analyzers reason about."""

from __future__ import annotations

COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

#: Motor vehicles, for the vehicle analyzer.
VEHICLE_CLASSES = frozenset({"car", "motorcycle", "bus", "truck", "bicycle"})
MOTOR_VEHICLE_CLASSES = frozenset({"car", "motorcycle", "bus", "truck"})

#: Things that indicate a person is or was present, beyond `person` itself.
PERSON_ADJACENT = frozenset({"person", "backpack", "handbag", "tie", "suitcase", "umbrella"})

ANIMAL_CLASSES = frozenset(
    {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
)
PET_CLASSES = frozenset({"cat", "dog", "bird"})

#: Devices that may be displaying content worth reading.
SCREEN_CLASSES = frozenset({"tv", "laptop", "cell phone"})

# -- scene inference -------------------------------------------------------
#
# Scene classification without a CLIP-sized model: the objects present are
# strong evidence about the setting. Weaker than a dedicated scene network,
# but it needs no extra weights, its reasoning is fully inspectable, and the
# evidence for every guess can be shown to the user. A CLIP-backed analyzer
# can be added later as a plugin without changing anything here.

SCENE_RULES: dict[str, dict[str, object]] = {
    "street / traffic": {
        "strong": {"traffic light", "stop sign", "fire hydrant", "bus", "parking meter"},
        "weak": {"car", "truck", "motorcycle", "bicycle", "person"},
        "detail": "Public roadway — often geolocatable from signage and street furniture",
    },
    "indoor / domestic": {
        "strong": {"bed", "couch", "toilet", "refrigerator", "microwave", "oven", "sink"},
        "weak": {"chair", "dining table", "tv", "potted plant", "vase", "book", "clock"},
        "detail": "Interior of a home",
    },
    "workplace / desk": {
        "strong": {"laptop", "keyboard", "mouse"},
        "weak": {"chair", "cup", "book", "cell phone", "tv"},
        "detail": "Desk or office setting — screens may carry readable content",
    },
    "food / hospitality": {
        "strong": {"pizza", "cake", "sandwich", "hot dog", "donut", "wine glass"},
        "weak": {"cup", "bowl", "fork", "knife", "spoon", "bottle", "dining table"},
        "detail": "Meal or venue setting",
    },
    "outdoor / nature": {
        "strong": {"bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"},
        "weak": {"potted plant", "bench", "kite", "frisbee"},
        "detail": "Open-air setting",
    },
    "sport / recreation": {
        "strong": {
            "sports ball", "skis", "snowboard", "surfboard", "tennis racket",
            "baseball bat", "baseball glove", "skateboard",
        },
        "weak": {"person", "bench"},
        "detail": "Sporting or recreational activity",
    },
    "transit / travel": {
        "strong": {"airplane", "train", "boat", "suitcase"},
        "weak": {"person", "backpack", "handbag", "bench"},
        "detail": "Travel context — stations, airports, ports",
    },
}


def infer_scenes(labels: list[str]) -> list[dict[str, object]]:
    """Rank candidate scenes from the object labels present.

    Returns entries with the evidence that produced them, so the reader can
    judge the guess rather than take it on faith.
    """
    present = set(labels)
    scored: list[dict[str, object]] = []

    for scene, rule in SCENE_RULES.items():
        strong = present & rule["strong"]  # type: ignore[operator]
        weak = present & rule["weak"]  # type: ignore[operator]
        if not strong and len(weak) < 2:
            continue
        # Strong cues are worth far more: a traffic light says "street"
        # much more decisively than a car, which could be anywhere.
        score = len(strong) * 1.0 + len(weak) * 0.25
        confidence = min(0.9, 0.35 + score * 0.12)
        scored.append(
            {
                "scene": scene,
                "score": round(score, 3),
                "confidence": round(confidence, 3),
                "evidence": sorted(strong | weak),
                "detail": rule["detail"],
            }
        )

    return sorted(scored, key=lambda s: -float(s["score"]))  # type: ignore[arg-type]
