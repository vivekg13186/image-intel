"""GPS extraction, offline reverse geocoding, and timestamp cross-checking.

The coordinates are the obvious part. The cross-check is the useful part: EXIF
``DateTimeOriginal`` is local wall-clock time while ``GPSTimeStamp`` is UTC, so
their difference implies a UTC offset. Comparing that against the timezone the
coordinates actually fall in catches fabricated locations and edited clocks.
"""

from __future__ import annotations

import re
from datetime import datetime
from functools import lru_cache
from typing import Any

from imgintel.core.analyzer import Analyzer, AnalyzerResult, Availability, Cost, Finding, Severity
from imgintel.core.context import AnalysisContext
from imgintel.core.deps import has_module


@lru_cache(maxsize=1)
def _timezone_finder():
    """Build the TimezoneFinder once per process.

    Constructing it loads several megabytes of boundary data — doing that per
    image cost ~340 ms and made the `quick` profile anything but.
    """
    from timezonefinder import TimezoneFinder  # noqa: PLC0415

    return TimezoneFinder()


@lru_cache(maxsize=4096)
def _lookup_place(lat_rounded: float, lon_rounded: float) -> dict | None:
    """Reverse geocode, memoized — batch runs often revisit the same spot."""
    import reverse_geocoder  # noqa: PLC0415

    hit = reverse_geocoder.search((lat_rounded, lon_rounded), mode=1, verbose=False)[0]
    return {
        "name": hit.get("name"),
        "admin1": hit.get("admin1"),
        "admin2": hit.get("admin2"),
        "country_code": hit.get("cc"),
    }


def _rational(value: Any) -> float | None:
    """Unwrap one component, which may be a number or a (numerator, denominator).

    Different readers hand back different shapes: exiftool gives floats, Pillow
    gives IFDRational (float-like), and raw EXIF parsers give integer pairs.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            num, den = float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
        return num / den if den else None
    return None


def _magnitude(value: Any) -> float | None:
    """Degrees from a signed decimal, a (d, m, s) triple, or nested rationals."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return abs(float(value))
    if isinstance(value, str):
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", value)]
        if not nums:
            return None
        value = nums
    if isinstance(value, (list, tuple)):
        # A bare (numerator, denominator) pair is a single value, not d/m.
        if len(value) == 2 and all(isinstance(v, int) for v in value) and value[1] not in (0, 1):
            single = _rational(value)
            return abs(single) if single is not None else None
        parts = [p for p in (_rational(v) for v in value[:3]) if p is not None]
        if not parts:
            return None
        parts += [0.0] * (3 - len(parts))
        return abs(parts[0]) + parts[1] / 60 + parts[2] / 3600
    return None


def _signed(value: Any, ref: Any, negative: str) -> float | None:
    mag = _magnitude(value)
    if mag is None:
        return None
    ref_s = str(ref or "").strip().upper()[:1]
    if ref_s == negative:
        return -mag
    if not ref_s and isinstance(value, (int, float)) and value < 0:
        return -mag
    return mag


def _dms(dec: float, positive: str, negative: str) -> str:
    hemi = positive if dec >= 0 else negative
    dec = abs(dec)
    d = int(dec)
    m = int((dec - d) * 60)
    s = (dec - d - m / 60) * 3600
    return f"{d}°{m:02d}'{s:05.2f}\"{hemi}"


class GpsAnalyzer(Analyzer):
    name = "gps"
    version = "1.0.0"
    title = "Location"
    description = "GPS coordinates, altitude, bearing, place name and timezone cross-check"
    requires = ("exif",)
    cost = Cost.CHEAP

    def available(self) -> Availability:
        return Availability.yes()  # degrades gracefully without the geo extras

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        tags: dict[str, Any] = ctx.data("exif").get("tags", {})
        gps = {k.split(":", 1)[-1]: v for k, v in tags.items() if re.match(r".*GPS", k, re.I)}
        if not gps:
            return self.ok({"present": False}, [])

        lat = _signed(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"), "S")
        lon = _signed(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"), "W")

        data: dict[str, Any] = {"present": True, "raw_tags": gps}
        findings: list[Finding] = []

        if lat is None or lon is None:
            findings.append(
                Finding(
                    "location",
                    "GPS tags present but coordinates incomplete",
                    sorted(gps),
                    Severity.MEDIUM,
                )
            )
            return self.ok(data, findings)

        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            findings.append(
                Finding("location", "Coordinates out of valid range", [lat, lon], Severity.HIGH)
            )
            return self.ok({**data, "latitude": lat, "longitude": lon}, findings)

        data.update(
            {
                "latitude": round(lat, 7),
                "longitude": round(lon, 7),
                "dms": f"{_dms(lat, 'N', 'S')} {_dms(lon, 'E', 'W')}",
                "google_maps": f"https://www.google.com/maps/search/?api=1&query={lat:.7f},{lon:.7f}",
                "openstreetmap": f"https://www.openstreetmap.org/?mlat={lat:.7f}&mlon={lon:.7f}#map=17/{lat:.5f}/{lon:.5f}",
                "geo_uri": f"geo:{lat:.7f},{lon:.7f}",
            }
        )

        if abs(lat) < 0.001 and abs(lon) < 0.001:
            findings.append(
                Finding(
                    "location",
                    "Null Island coordinates (0, 0)",
                    [lat, lon],
                    Severity.HIGH,
                    detail="Almost always a default, placeholder or corrupted value",
                )
            )
        else:
            findings.append(
                Finding(
                    "location",
                    "GPS coordinates",
                    f"{lat:.6f}, {lon:.6f}",
                    Severity.HIGH,
                    detail=data["dms"],
                )
            )

        self._altitude(gps, data, findings)
        self._bearing(gps, data, findings)
        self._speed(gps, data)
        self._accuracy(gps, data, findings)
        self._place(lat, lon, data, findings)
        self._timecheck(ctx, gps, lat, lon, data, findings)

        return self.ok(data, findings)

    # -- components --------------------------------------------------------

    def _altitude(self, gps: dict, data: dict, findings: list[Finding]) -> None:
        alt = gps.get("GPSAltitude")
        if alt is None:
            return
        value = _magnitude(alt)
        if value is None:
            return
        ref = gps.get("GPSAltitudeRef")
        below = str(ref).strip() in ("1", "Below Sea Level") or (
            isinstance(alt, (int, float)) and alt < 0
        )
        metres = -value if below else value
        data["altitude_m"] = round(metres, 2)
        findings.append(
            Finding("location", "Altitude", f"{metres:.1f} m", Severity.INFO)
        )

    def _bearing(self, gps: dict, data: dict, findings: list[Finding]) -> None:
        direction = _magnitude(gps.get("GPSImgDirection"))
        if direction is None:
            return
        ref = str(gps.get("GPSImgDirectionRef") or "").strip().upper()[:1]
        data["img_direction_deg"] = round(direction, 2)
        data["img_direction_ref"] = {"T": "true north", "M": "magnetic north"}.get(ref, ref or None)
        compass = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((direction + 22.5) % 360 // 45)]
        findings.append(
            Finding(
                "location",
                "Camera bearing",
                f"{direction:.0f}° ({compass})",
                Severity.INFO,
                detail="Direction the lens was pointing — narrows down the exact vantage point",
            )
        )

    def _speed(self, gps: dict, data: dict) -> None:
        speed = _magnitude(gps.get("GPSSpeed"))
        if speed is None:
            return
        unit = {"K": "km/h", "M": "mph", "N": "knots"}.get(
            str(gps.get("GPSSpeedRef") or "").strip().upper()[:1], "unknown"
        )
        data["speed"] = {"value": round(speed, 3), "unit": unit}

    def _accuracy(self, gps: dict, data: dict, findings: list[Finding]) -> None:
        dop = _magnitude(gps.get("GPSDOP"))
        if dop is None:
            return
        data["dop"] = round(dop, 3)
        if dop > 5:
            findings.append(
                Finding(
                    "location",
                    "Poor GPS fix quality",
                    f"DOP {dop:.1f}",
                    Severity.LOW,
                    detail="Dilution of precision above 5 — coordinates may be tens of metres off",
                )
            )

    def _place(self, lat: float, lon: float, data: dict, findings: list[Finding]) -> None:
        if not has_module("reverse_geocoder"):
            data["place"] = None
            data["place_note"] = "install imgintel[places] for offline place names"
            return
        try:
            place = _lookup_place(round(lat, 4), round(lon, 4))
            data["place"] = place
            if place is None:
                return
            label = ", ".join(
                str(v) for v in (place["name"], place["admin1"], place["country_code"]) if v
            )
            findings.append(
                Finding(
                    "location",
                    "Nearest known place",
                    label,
                    Severity.MEDIUM,
                    confidence=0.85,
                    detail="Nearest populated place from an offline gazetteer, not a street address",
                )
            )
        except Exception as exc:  # noqa: BLE001
            data["place"] = None
            data["place_error"] = str(exc)

    def _timecheck(
        self,
        ctx: AnalysisContext,
        gps: dict,
        lat: float,
        lon: float,
        data: dict,
        findings: list[Finding],
    ) -> None:
        """Compare the offset implied by EXIF-vs-GPS time against the real timezone."""
        tz_name = None
        if has_module("timezonefinder"):
            try:
                tz_name = _timezone_finder().timezone_at(lat=lat, lng=lon)
                data["timezone"] = tz_name
            except Exception as exc:  # noqa: BLE001
                data["timezone_error"] = str(exc)
        else:
            data["timezone_note"] = "install imgintel[timezone] for timezone cross-checking"

        local_str = ctx.data("exif").get("highlights", {}).get("datetime_original")
        gps_date = gps.get("GPSDateStamp")
        gps_time = gps.get("GPSTimeStamp")
        if not (local_str and gps_date and gps_time):
            return

        local = _parse_exif_dt(str(local_str))
        utc = _parse_gps_dt(str(gps_date), gps_time)
        if not (local and utc):
            return

        offset_hours = (local - utc).total_seconds() / 3600
        data["implied_utc_offset_hours"] = round(offset_hours, 2)
        data["gps_timestamp_utc"] = utc.isoformat()

        if abs(offset_hours) > 14.5:
            findings.append(
                Finding(
                    "timeline",
                    "Implied UTC offset is impossible",
                    f"{offset_hours:+.1f} h",
                    Severity.HIGH,
                    confidence=0.9,
                    detail="EXIF local time and GPS UTC time disagree by more than any real "
                    "timezone — one of the two has been altered",
                )
            )
            return

        if tz_name:
            try:
                from zoneinfo import ZoneInfo  # noqa: PLC0415

                actual = utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo(tz_name))
                real_offset = actual.utcoffset().total_seconds() / 3600  # type: ignore[union-attr]
                data["timezone_offset_hours"] = round(real_offset, 2)
                drift = abs(offset_hours - real_offset)
                if drift > 1.0:
                    findings.append(
                        Finding(
                            "timeline",
                            "Timestamp inconsistent with GPS timezone",
                            f"implied {offset_hours:+.1f} h, {tz_name} is {real_offset:+.1f} h",
                            Severity.MEDIUM,
                            confidence=0.75,
                            detail="Camera clock set to a different timezone, or the location "
                            "or timestamp was edited",
                        )
                    )
                else:
                    findings.append(
                        Finding(
                            "timeline",
                            "Timestamp consistent with GPS timezone",
                            tz_name,
                            Severity.INFO,
                            confidence=0.8,
                        )
                    )
            except Exception:  # noqa: BLE001
                pass


def _parse_exif_dt(text: str) -> datetime | None:
    text = text.strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(text[:26], fmt)
        except ValueError:
            continue
    return None


def _parse_gps_dt(date_text: str, time_value: Any) -> datetime | None:
    date_text = date_text.strip().replace("-", ":")[:10]
    try:
        base = datetime.strptime(date_text, "%Y:%m:%d")
    except ValueError:
        return None
    if isinstance(time_value, (list, tuple)):
        parts = [float(v) for v in time_value[:3] if isinstance(v, (int, float))]
    else:
        parts = [float(x) for x in re.findall(r"\d+\.?\d*", str(time_value))[:3]]
    if not parts:
        return None
    parts += [0.0] * (3 - len(parts))
    return base.replace(
        hour=int(parts[0]) % 24, minute=int(parts[1]) % 60, second=int(parts[2]) % 60
    )
