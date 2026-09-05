"""Face **identification** — matching detected faces against enrolled identities.

This is the one analyzer in imgintel that processes biometric data, and it is
the only one that refuses to run rather than degrading. Two independent
conditions must both hold:

1. a lawful basis recorded in the case file, and
2. ``--allow-biometrics`` on this specific run.

The split is deliberate. The case file answers "were you entitled to do this"
and travels with the evidence; the run flag answers "did you mean to do it
now", and must never be inherited silently from a config file someone set up
months ago. When either is missing the analyzer reports ``skipped`` with the
reason, which is then visible in the report and the evidence bundle — so the
record shows that identification was *not* performed, and why.

Every verdict carries a margin. See :mod:`imgintel.store.entities` for why an
identification tool that cannot return "inconclusive" is not usable evidence.
"""

from __future__ import annotations

from typing import Any

from imgintel.core.analyzer import Analyzer, AnalyzerResult, Availability, Cost, Finding, Severity
from imgintel.core.context import AnalysisContext
from imgintel.core.deps import has_module
from imgintel.util.embed import FACE_EMBEDDER, embed_faces


class FaceIdentityAnalyzer(Analyzer):
    name = "facedb"
    version = "1.0.0"
    title = "Face identification"
    description = "Matches detected faces against enrolled identities (gated: biometric data)"
    requires = ("faces",)
    cost = Cost.HEAVY
    needs_models = ("face-embed",)

    def precheck(self, ctx: AnalysisContext) -> str | None:
        """The gate. Runs before dependencies, models, or any pixel is touched.

        Deliberately first: if there is no authority to identify anyone, that
        is the reason to record — not that the embedding model happens to be
        missing, which would be true but beside the point.
        """
        refusal = ctx.case.biometric_refusal(run_flag=ctx.allow_biometrics)
        if refusal:
            return refusal
        if ctx.entity_db is None:
            return "no entity database selected (pass --entity-db)"
        if not ctx.entity_db.exists():
            return f"entity database not found: {ctx.entity_db}"
        return None

    def available(self) -> Availability:
        if not has_module("cv2"):
            return Availability.no("opencv is not installed", "pip install 'imgintel[faces]'")
        import cv2  # noqa: PLC0415

        if not hasattr(cv2, "FaceRecognizerSF"):
            return Availability.no(
                f"opencv {cv2.__version__} has no FaceRecognizerSF",
                "pip install -U opencv-python-headless",
            )
        from imgintel.core.modelstore import available as models_available

        ok, reason = models_available(*self.needs_models)
        if ok:
            return Availability.yes()
        # The hint is what makes an unavailable analyzer actionable.
        return Availability.no(reason, f"imgintel models pull {' '.join(self.needs_models)}")

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        # The gate has already refused in precheck() if it was going to; no
        # biometric template is ever computed before that check passes.
        faces = ctx.data("faces").get("faces", [])
        if not faces:
            return self.ok(
                {"searched": 0, "matches": [], "identified": 0, "database": str(ctx.entity_db)},
                [],
            )

        from imgintel.core.modelstore import status as model_status
        from imgintel.store.entities import (
            DEFAULT_MARGIN,
            DEFAULT_THRESHOLD,
            EntityStore,
        )

        model = model_status("face-embed")
        # Detected at analysis scale, embedded at full resolution — see
        # embed_faces for why the extra decode earns its keep here.
        full_height, full_width = ctx.rgb.shape[:2]
        small_width = ctx.small.shape[1]
        scale = full_width / small_width if small_width else 1.0
        vectors = embed_faces(ctx.rgb, faces, model.path, scale=scale)

        results: list[dict[str, Any]] = []
        findings: list[Finding] = []
        identified = 0

        with EntityStore(ctx.entity_db) as store:
            enrolled = store.count("person")
            store.log(
                "search:person",
                f"{len(faces)} face(s) against {enrolled} enrolled identities",
                operator=ctx.case.operator,
                case_id=ctx.case.case_id,
            )

            for face, vector in zip(faces, vectors, strict=True):
                if vector is None:
                    results.append({"bbox": face["bbox"], "error": "alignment failed"})
                    continue
                try:
                    match, shortlist = store.search(
                        vector,
                        kind="person",
                        embedder=FACE_EMBEDDER,
                        threshold=DEFAULT_THRESHOLD,
                        margin=DEFAULT_MARGIN,
                    )
                except ValueError as exc:
                    return self.skipped(str(exc))

                entry: dict[str, Any] = {
                    "bbox": face["bbox"],
                    "detection_confidence": face["confidence"],
                    "candidates": shortlist,
                }
                if match is not None:
                    entry.update(match.as_dict())
                    if match.conclusive:
                        identified += 1
                    findings.extend(_match_findings(match, face))
                else:
                    entry["verdict"] = "no candidates"
                results.append(entry)

        data = {
            "database": str(ctx.entity_db),
            "embedder": FACE_EMBEDDER,
            "model_sha256": model.sha256,
            "enrolled_identities": enrolled,
            "searched": len(faces),
            "identified": identified,
            "threshold": DEFAULT_THRESHOLD,
            "margin": DEFAULT_MARGIN,
            "embedded_at": [full_width, full_height],
            "lawful_basis": ctx.case.biometric_lawful_basis,
            "authorised_by": ctx.case.biometric_authorised_by,
            "matches": results,
        }

        if not findings:
            findings.append(
                Finding(
                    "identity",
                    "No enrolled identity matched",
                    f"{len(faces)} face(s) searched against {enrolled} identities",
                    Severity.INFO,
                    detail="Absence of a match is not evidence the person is not enrolled — "
                    "pose, resolution and occlusion all suppress recognition",
                )
            )
        return self.ok(data, findings)


def _match_findings(match, face: dict) -> list[Finding]:
    bbox = tuple(face["bbox"])

    if match.verdict == "match":
        return [
            Finding(
                "identity",
                f"Identity match: {match.entity}",
                f"similarity {match.similarity:.3f}, margin {match.margin:.3f}",
                Severity.HIGH,
                confidence=round(min(0.95, match.similarity), 3),
                bbox=bbox,
                detail="Exceeds both the similarity threshold and the margin over the next "
                "candidate. A biometric match is investigative lead material, not proof of "
                "identity — corroborate before relying on it",
            )
        ]

    if match.verdict == "inconclusive":
        return [
            Finding(
                "identity",
                "Inconclusive identity match",
                f"{match.entity} ({match.similarity:.3f}) vs "
                f"{match.runner_up} ({match.runner_up_similarity:.3f})",
                Severity.MEDIUM,
                confidence=0.4,
                bbox=bbox,
                detail="Above the similarity threshold but too close to the next candidate to "
                "distinguish them. Reported deliberately rather than resolved by picking the "
                "higher score",
            )
        ]

    return []
