"""Scene classification from object evidence.

A CLIP or Places365 model would classify scenes directly and more accurately.
This does it from what the object detector already found, which trades some
accuracy for three things that matter more here:

* **No extra weights.** CLIP is ~350 MB plus a tokenizer; this is a lookup table.
* **Inspectable reasoning.** Every guess ships the objects that produced it, so
  a reader can judge it rather than trust a softmax.
* **User-extensible.** The rules live in ``imgintel.util.coco.SCENE_RULES`` and
  are editable without retraining anything.

A CLIP-backed scene analyzer can be added later as a plugin; because it would
register under a different name, both can run and be compared.
"""

from __future__ import annotations

from imgintel.core.analyzer import Analyzer, AnalyzerResult, Cost, Finding, Severity
from imgintel.core.context import AnalysisContext
from imgintel.util.coco import infer_scenes

#: Below this the evidence is too thin to name a scene at all.
MIN_CONFIDENCE = 0.45


class SceneAnalyzer(Analyzer):
    name = "scene"
    version = "1.0.0"
    title = "Scene"
    description = "Infers the setting from detected objects, with its evidence"
    requires = ("objects",)
    cost = Cost.CHEAP

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        labels = [d["label"] for d in ctx.data("objects").get("detections", [])]
        scenes = infer_scenes(labels)

        data = {
            "method": "object-evidence rules",
            "object_labels": sorted(set(labels)),
            "candidates": scenes,
            "primary": scenes[0]["scene"] if scenes else None,
        }

        if not scenes:
            return self.ok(
                data,
                [
                    Finding(
                        "scene",
                        "Scene not determined",
                        None,
                        Severity.INFO,
                        detail="Too few recognisable objects to infer a setting",
                    )
                ],
            )

        top = scenes[0]
        findings = [
            Finding(
                "scene",
                "Likely scene",
                top["scene"],
                Severity.INFO,
                confidence=float(top["confidence"]),
                detail=f"{top['detail']}. Evidence: {', '.join(top['evidence'])}",
            )
        ]

        # A second candidate that is nearly as well supported is worth naming:
        # it stops a marginal call being read as a firm one.
        if len(scenes) > 1 and float(scenes[1]["score"]) >= float(top["score"]) * 0.75:
            findings.append(
                Finding(
                    "scene",
                    "Alternative scene reading",
                    scenes[1]["scene"],
                    Severity.INFO,
                    confidence=float(scenes[1]["confidence"]),
                    detail=f"Evidence: {', '.join(scenes[1]['evidence'])}",
                )
            )

        if top["scene"] == "street / traffic" and float(top["confidence"]) >= MIN_CONFIDENCE:
            findings.append(
                Finding(
                    "location",
                    "Outdoor public scene",
                    top["scene"],
                    Severity.MEDIUM,
                    confidence=float(top["confidence"]),
                    detail="Street furniture, signage and business names in frame are often "
                    "enough to geolocate the image",
                )
            )

        return self.ok(data, findings)
