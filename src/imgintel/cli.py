"""Command-line interface.

Deliberately thin: every command resolves options and hands off to
``imgintel.core.engine``. Nothing here knows how any analyzer works.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from imgintel import __version__
from imgintel.core.engine import SelectionError, analyze_image, plan
from imgintel.core.profiles import DEFAULT_PROFILE, PROFILES
from imgintel.core.registry import Registry, user_plugin_dir
from imgintel.core.schema import SCHEMA_VERSION, FindingsDocument
from imgintel.plugins import API_VERSION as PLUGIN_API_VERSION
from imgintel.report import (
    export_bundle,
    render_html,
    verify_bundle,
    write_findings_csv,
    write_summary_csv,
)

app = typer.Typer(
    name="imgintel",
    help="Extract intelligence from images.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()
err_console = Console(stderr=True)

SEVERITY_STYLE = {
    "high": "bold red",
    "medium": "yellow",
    "low": "cyan",
    "info": "dim white",
}
STATUS_STYLE = {
    "ok": "green",
    "skipped": "dim",
    "unavailable": "yellow",
    "error": "red",
}


def _version_callback(value: bool) -> None:
    if value:
        console.print(
            f"imgintel {__version__} (schema {SCHEMA_VERSION}, "
            f"plugin API {PLUGIN_API_VERSION})"
        )
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option("--version", "-V", callback=_version_callback, is_eager=True,
                     help="Show version and exit."),
    ] = False,
) -> None:
    pass


# ---------------------------------------------------------------- analyze --


@app.command()
def analyze(
    image: Annotated[Path, typer.Argument(help="Image file to analyze.")],
    profile: Annotated[
        str, typer.Option("--profile", "-p", help=f"One of: {', '.join(PROFILES)}.")
    ] = DEFAULT_PROFILE,
    analyzers: Annotated[
        str | None,
        typer.Option("--analyzers", "-a", help="Comma-separated analyzers; overrides --profile."),
    ] = None,
    exclude: Annotated[
        str | None, typer.Option("--exclude", "-x", help="Comma-separated analyzers to skip.")
    ] = None,
    json_out: Annotated[
        Path | None, typer.Option("--json", "-j", help="Write the findings document to this file.")
    ] = None,
    html_out: Annotated[
        Path | None, typer.Option("--html", help="Write a self-contained HTML report.")
    ] = None,
    csv_out: Annotated[
        Path | None, typer.Option("--csv", help="Write findings as CSV (one row per finding).")
    ] = None,
    evidence_out: Annotated[
        Path | None,
        typer.Option("--evidence", help="Write a hashed evidence bundle into this directory."),
    ] = None,
    stdout_json: Annotated[
        bool, typer.Option("--stdout-json", help="Print the full JSON document to stdout.")
    ] = False,
    show_data: Annotated[
        bool, typer.Option("--show-data", help="Also print each analyzer's raw data block.")
    ] = False,
    allow_network: Annotated[
        bool, typer.Option("--allow-network", help="Permit analyzers that contact the network.")
    ] = False,
    allow_biometrics: Annotated[
        bool,
        typer.Option(
            "--allow-biometrics",
            help="Permit face identification. Also needs a lawful basis in the case file.",
        ),
    ] = False,
    entity_db: Annotated[
        Path | None, typer.Option("--entity-db", help="Entity database to match against.")
    ] = None,
    allow_local_plugins: Annotated[
        bool,
        typer.Option("--allow-local-plugins", help=f"Load .py plugins from {user_plugin_dir()}."),
    ] = False,
    case_id: Annotated[str | None, typer.Option("--case", help="Case identifier.")] = None,
    operator: Annotated[str | None, typer.Option("--operator", help="Operator name.")] = None,
    case_file: Annotated[
        Path | None, typer.Option("--case-file", help="Case file with the lawful basis.")
    ] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress the report.")] = False,
) -> None:
    """Analyze a single image."""
    from imgintel.core import casefile

    registry = Registry.discover(allow_local=allow_local_plugins)
    _warn_load_errors(registry)

    case = casefile.merge(
        casefile.load(case_file or casefile.default_path()),
        case_id=case_id,
        operator=operator,
    )

    try:
        doc = analyze_image(
            image,
            registry=registry,
            profile=profile,
            only=_split(analyzers),
            exclude=_split(exclude),
            allow_network=allow_network,
            allow_biometrics=allow_biometrics,
            entity_db=entity_db,
            case=case,
            command=shlex.join(sys.argv),
        )
    except (SelectionError, KeyError) as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(2) from exc
    except FileNotFoundError:
        err_console.print(f"[red]Error:[/red] no such file: {image}")
        raise typer.Exit(2) from None
    except IsADirectoryError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(2) from exc

    written: list[Path] = []
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        written.append(json_out)
    if html_out:
        html_out.parent.mkdir(parents=True, exist_ok=True)
        html_out.write_text(render_html(doc, image), encoding="utf-8")
        written.append(html_out)
    if csv_out:
        write_findings_csv([doc], csv_out)
        written.append(csv_out)
    if evidence_out:
        bundle = export_bundle(doc, evidence_out, image_path=image)
        written.append(bundle.directory)

    if written and not quiet and not stdout_json:
        for path in written:
            console.print(f"[green]Wrote[/green] {path}")

    if stdout_json:
        sys.stdout.write(doc.model_dump_json(indent=2) + "\n")
    elif not quiet:
        _render(doc, show_data=show_data)

    # Exit 1 when any analyzer errored, so scripts can detect partial results.
    raise typer.Exit(1 if doc.summary.analyzers_failed else 0)


# ------------------------------------------------------------------ batch --


@app.command()
def batch(
    directory: Annotated[Path, typer.Argument(help="Directory (or single file) to analyze.")],
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Output directory for results.")
    ] = Path("imgintel-results"),
    profile: Annotated[str, typer.Option("--profile", "-p")] = DEFAULT_PROFILE,
    analyzers: Annotated[str | None, typer.Option("--analyzers", "-a")] = None,
    exclude: Annotated[str | None, typer.Option("--exclude", "-x")] = None,
    recursive: Annotated[
        bool, typer.Option("--recursive/--no-recursive", "-r", help="Descend into subdirectories.")
    ] = True,
    workers: Annotated[
        int, typer.Option("--workers", "-w", help="Worker processes (0 = auto).")
    ] = 0,
    resume: Annotated[
        bool, typer.Option("--resume", help="Skip images already recorded as done.")
    ] = False,
    write_json: Annotated[
        bool, typer.Option("--write-json", help="Also write one JSON document per image.")
    ] = False,
    html_index: Annotated[
        bool, typer.Option("--html/--no-html", help="Write per-image HTML reports.")
    ] = False,
    allow_network: Annotated[bool, typer.Option("--allow-network")] = False,
    allow_biometrics: Annotated[
        bool, typer.Option("--allow-biometrics", help="Permit face identification.")
    ] = False,
    entity_db: Annotated[
        Path | None, typer.Option("--entity-db", help="Entity database to match against.")
    ] = None,
    allow_local_plugins: Annotated[bool, typer.Option("--allow-local-plugins")] = False,
    case_id: Annotated[str | None, typer.Option("--case")] = None,
    operator: Annotated[str | None, typer.Option("--operator")] = None,
    case_file: Annotated[Path | None, typer.Option("--case-file")] = None,
    limit: Annotated[
        int, typer.Option("--limit", help="Stop after this many images (0 = no limit).")
    ] = 0,
) -> None:
    """Analyze every image under a directory."""
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    from imgintel.runner import BatchConfig, find_images, run_batch
    from imgintel.store.db import open_index

    if not directory.exists():
        err_console.print(f"[red]Error:[/red] no such path: {directory}")
        raise typer.Exit(2)

    images = find_images(directory, recursive=recursive)
    if limit > 0:
        images = images[:limit]
    if not images:
        err_console.print(f"[yellow]No images found under[/yellow] {directory}")
        raise typer.Exit(2)

    out.mkdir(parents=True, exist_ok=True)
    index = open_index(out)
    done = index.completed() if resume else set()

    from imgintel.core import casefile

    config = BatchConfig(
        profile=profile,
        only=_split(analyzers),
        exclude=_split(exclude),
        allow_network=allow_network,
        allow_biometrics=allow_biometrics,
        entity_db=str(entity_db) if entity_db else None,
        allow_local_plugins=allow_local_plugins,
        case=casefile.merge(
            casefile.load(case_file or casefile.default_path()),
            case_id=case_id,
            operator=operator,
        ),
        workers=workers,
        write_json=write_json,
    )

    json_dir = out / "json"
    html_dir = out / "html"
    if write_json:
        json_dir.mkdir(exist_ok=True)
    if html_index:
        html_dir.mkdir(exist_ok=True)

    marks: list[tuple[str, str, str | None]] = []
    columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ]

    with Progress(*columns, console=console) as progress:
        task = progress.add_task("Analyzing", total=len(images) - len(done))

        def on_result(path: str, doc: FindingsDocument | None, error: str | None) -> None:
            progress.advance(task)
            marks.append((path, "failed" if error else "done", error))
            if doc is None:
                return
            stem = Path(path).stem
            if write_json:
                (json_dir / f"{stem}.json").write_text(
                    doc.model_dump_json(indent=2), encoding="utf-8"
                )
            if html_index:
                (html_dir / f"{stem}.html").write_text(render_html(doc, path), encoding="utf-8")

        result = run_batch(
            images, config, already_done=done, on_result=on_result, command=shlex.join(sys.argv)
        )

    index.mark_many(marks)
    index.upsert_many(result.rows)

    write_summary_csv(result.rows, out / "summary.csv")
    write_findings_csv(result.documents, out / "findings.csv")
    (out / "findings.jsonl").write_text(
        "\n".join(d.model_dump_json() for d in result.documents) + "\n", encoding="utf-8"
    )

    stats = index.stats()
    index.close()

    console.print(
        Panel(
            f"[bold]{result.analyzed}[/bold] analyzed · "
            f"[red]{len(result.failures)}[/red] failed · "
            f"[dim]{result.skipped} skipped[/dim] · {result.workers} worker(s)\n"
            f"{int(stats['high'] or 0)} high-severity findings across "
            f"{stats['images']} indexed images · {stats['with_gps']} with GPS\n"
            f"[dim]{out}/summary.csv · findings.csv · findings.jsonl · index.db[/dim]",
            title="batch complete",
            border_style="blue",
        )
    )
    for path, error in result.failures[:10]:
        err_console.print(f"[red]failed[/red] {path}: {error}")

    raise typer.Exit(1 if result.failures else 0)


# ---------------------------------------------------------------- compare --


@app.command()
def compare(
    first: Annotated[Path, typer.Argument(help="First image.")],
    second: Annotated[Path, typer.Argument(help="Second image.")],
) -> None:
    """Compare two images by file, pixel and perceptual hashes."""
    from imgintel.util.imgmath import hamming_hex

    registry = Registry.discover()
    docs = []
    for path in (first, second):
        if not path.exists():
            err_console.print(f"[red]Error:[/red] no such file: {path}")
            raise typer.Exit(2)
        docs.append(
            analyze_image(path, registry=registry, only=["hashes", "perceptual", "fileinfo"])
        )

    a, b = (d.analyzers for d in docs)
    table = Table(header_style="bold", expand=True)
    table.add_column("Measure", style="cyan", no_wrap=True)
    table.add_column(first.name, overflow="fold")
    table.add_column(second.name, overflow="fold")
    table.add_column("Verdict", no_wrap=True)

    same_file = a["hashes"].data["sha256"] == b["hashes"].data["sha256"]
    same_pixels = a["perceptual"].data["pixel_sha256"] == b["perceptual"].data["pixel_sha256"]
    table.add_row(
        "SHA-256",
        a["hashes"].data["sha256"][:24] + "…",
        b["hashes"].data["sha256"][:24] + "…",
        Text("identical", style="green") if same_file else Text("differ", style="yellow"),
    )
    table.add_row(
        "Pixel SHA-256",
        a["perceptual"].data["pixel_sha256"][:24] + "…",
        b["perceptual"].data["pixel_sha256"][:24] + "…",
        Text("identical", style="green") if same_pixels else Text("differ", style="yellow"),
    )

    distances = {}
    for name in ("phash", "dhash", "ahash"):
        ha, hb = a["perceptual"].data[name], b["perceptual"].data[name]
        distance = hamming_hex(ha, hb)
        distances[name] = distance
        bits = len(ha) * 4
        table.add_row(name, ha, hb, f"{distance}/{bits} bits differ")

    console.print(table)

    # Each hash fails on a different transform, so the verdict follows the
    # closest of them rather than pHash alone: a plain resize of a
    # high-frequency image can sit 16 bits away in pHash and 4 in dHash.
    closest = min(distances, key=lambda k: distances[k])
    best = distances[closest]

    if same_file:
        verdict, style = "Identical files.", "green"
    elif same_pixels:
        verdict, style = (
            "Same pixels, different container — metadata was added, removed or edited.",
            "yellow",
        )
    elif best <= 4:
        verdict, style = (
            f"Near-identical: almost certainly the same image, re-encoded or resized "
            f"({closest} distance {best}).",
            "yellow",
        )
    elif best <= 10:
        verdict, style = (
            f"Similar: likely the same scene, possibly cropped or edited "
            f"({closest} distance {best}).",
            "yellow",
        )
    else:
        verdict, style = f"Different images (closest hash {closest} at {best} bits).", "dim"
    console.print(Panel(verdict, border_style=style))


# ----------------------------------------------------------------- dedupe --


@app.command()
def dedupe(
    target: Annotated[Path, typer.Argument(help="Directory of images, or an existing index.db.")],
    threshold: Annotated[
        int, typer.Option("--threshold", "-t", help="Max Hamming distance for a near-duplicate.")
    ] = 8,
    recursive: Annotated[bool, typer.Option("--recursive/--no-recursive", "-r")] = True,
    workers: Annotated[int, typer.Option("--workers", "-w")] = 0,
    csv_out: Annotated[
        Path | None, typer.Option("--csv", help="Write clusters to CSV.")
    ] = None,
) -> None:
    """Find exact and near-duplicate images."""
    from imgintel.runner import BatchConfig, find_images, run_batch
    from imgintel.store.bktree import cluster_multi
    from imgintel.store.db import ImageIndex

    if target.is_file() and target.suffix == ".db":
        index = ImageIndex(target)
        rows = list(index.rows())
        exact = index.exact_duplicates("sha256")
        pixel_dupes = index.exact_duplicates("pixel_sha256")
        index.close()
    else:
        images = find_images(target, recursive=recursive)
        if not images:
            err_console.print(f"[yellow]No images found under[/yellow] {target}")
            raise typer.Exit(2)
        console.print(f"Hashing {len(images)} images…")
        result = run_batch(
            images,
            BatchConfig(only=["hashes", "perceptual", "fileinfo"], workers=workers),
            command=shlex.join(sys.argv),
        )
        rows = result.rows
        exact = _group(rows, "sha256")
        pixel_dupes = _group(rows, "pixel_sha256")

    console.print(f"[dim]{len(rows)} images indexed[/dim]")

    _dupe_table("Identical files (same SHA-256)", exact, "green")
    only_pixel = [(d, p) for d, p in pixel_dupes if p not in [x[1] for x in exact]]
    _dupe_table(
        "Identical pixels, different file (metadata differs)", only_pixel, "yellow"
    )

    # Link on any perceptual hash, not just pHash — see cluster_multi.
    usable = [r for r in rows if r.get("phash") or r.get("dhash")]
    clusters = cluster_multi(
        [[r.get("phash"), r.get("dhash")] for r in usable],
        [r["path"] for r in usable],
        threshold,
    )
    # Drop clusters already fully explained by exact duplication.
    exact_paths = {p for _, paths in exact for p in paths}
    near = [c for c in clusters if not {p for _, p in c} <= exact_paths]

    if near:
        console.print(
            f"\n[bold]Near-duplicates[/bold] "
            f"(pHash or dHash distance ≤ {threshold}; single-linkage clusters)"
        )
        for i, group in enumerate(near, 1):
            table = Table(show_header=False, expand=True, box=None, padding=(0, 1))
            for distance, path in group:
                label = "[dim]reference[/dim]" if distance == 0 else f"{distance} bits"
                table.add_row(label, str(path))
            console.print(Panel(table, title=f"cluster {i} · {len(group)} images",
                                border_style="yellow"))
    else:
        console.print("\n[dim]No near-duplicates found.[/dim]")

    if csv_out:
        import csv as csv_mod

        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with csv_out.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv_mod.writer(fh)
            writer.writerow(["cluster", "distance", "path"])
            for i, group in enumerate(near, 1):
                for distance, path in group:
                    writer.writerow([i, distance, path])
        console.print(f"[green]Wrote[/green] {csv_out}")


def _group(rows: list[dict], column: str) -> list[tuple[str, list[str]]]:
    buckets: dict[str, list[str]] = {}
    for row in rows:
        value = row.get(column)
        if value:
            buckets.setdefault(value, []).append(row["path"])
    return [(digest, paths) for digest, paths in buckets.items() if len(paths) > 1]


def _dupe_table(title: str, groups: list, style: str) -> None:
    if not groups:
        return
    console.print(f"\n[bold]{title}[/bold]")
    for digest, paths in groups:
        table = Table(show_header=False, expand=True, box=None, padding=(0, 1))
        for path in paths:
            table.add_row(str(path))
        console.print(Panel(table, title=f"{digest[:16]}… · {len(paths)} files",
                            border_style=style))


# --------------------------------------------------------------- evidence --

evidence_app = typer.Typer(help="Create and verify evidence bundles.", no_args_is_help=True)
app.add_typer(evidence_app, name="evidence")


@evidence_app.command("export")
def evidence_export(
    findings: Annotated[Path, typer.Argument(help="A findings.json written by `analyze`.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Directory to create the bundle in.")]
    = Path("evidence"),
    image: Annotated[
        Path | None, typer.Option("--image", help="Override the original image path.")
    ] = None,
) -> None:
    """Build a hashed, self-describing evidence bundle from a findings document."""
    if not findings.exists():
        err_console.print(f"[red]Error:[/red] no such file: {findings}")
        raise typer.Exit(2)
    doc = FindingsDocument.model_validate_json(findings.read_text(encoding="utf-8"))
    bundle = export_bundle(doc, out, image_path=image)

    table = Table(header_style="bold", expand=True)
    table.add_column("File", style="cyan")
    for name in [*bundle.files, "MANIFEST.sha256"]:
        table.add_row(name)
    console.print(Panel(table, title=str(bundle.directory), border_style="green"))
    console.print("[dim]Verify later with:[/dim] "
                  f"imgintel evidence verify {bundle.directory}")


@evidence_app.command("verify")
def evidence_verify(
    bundle: Annotated[Path, typer.Argument(help="An evidence bundle directory.")],
) -> None:
    """Re-hash every file in a bundle and report anything that changed."""
    if not bundle.is_dir():
        err_console.print(f"[red]Error:[/red] not a directory: {bundle}")
        raise typer.Exit(2)

    result = verify_bundle(bundle)
    if result.ok:
        console.print(Panel(f"[green]{result.summary}[/green]", title=str(bundle),
                            border_style="green"))
        raise typer.Exit(0)

    lines = []
    for label, items in (
        ("MODIFIED", result.modified),
        ("MISSING", result.missing),
        ("NOT IN MANIFEST", result.unlisted),
    ):
        lines.extend(f"[red]{label}[/red] {item}" for item in items)
    console.print(Panel("\n".join(lines) or result.summary, title=f"{bundle} — INTEGRITY FAILURE",
                        border_style="red"))
    raise typer.Exit(1)


# ----------------------------------------------------------------- report --


@app.command()
def report(
    findings: Annotated[Path, typer.Argument(help="A findings.json written by `analyze`.")],
    html_out: Annotated[
        Path | None, typer.Option("--html", help="Write an HTML report here.")
    ] = None,
    csv_out: Annotated[Path | None, typer.Option("--csv", help="Write findings CSV here.")] = None,
    image: Annotated[
        Path | None, typer.Option("--image", help="Image to embed (defaults to the recorded path).")
    ] = None,
) -> None:
    """Re-render an existing findings document without re-analyzing."""
    if not findings.exists():
        err_console.print(f"[red]Error:[/red] no such file: {findings}")
        raise typer.Exit(2)
    doc = FindingsDocument.model_validate_json(findings.read_text(encoding="utf-8"))

    if not html_out and not csv_out:
        _render(doc)
        return
    if html_out:
        html_out.parent.mkdir(parents=True, exist_ok=True)
        html_out.write_text(render_html(doc, image), encoding="utf-8")
        console.print(f"[green]Wrote[/green] {html_out}")
    if csv_out:
        write_findings_csv([doc], csv_out)
        console.print(f"[green]Wrote[/green] {csv_out}")


# ------------------------------------------------------------------- case --

case_app = typer.Typer(help="Case metadata and biometric authorisation.", no_args_is_help=True)
app.add_typer(case_app, name="case")


@case_app.command("set")
def case_set(
    case_id: Annotated[str | None, typer.Option("--case-id")] = None,
    operator: Annotated[str | None, typer.Option("--operator")] = None,
    notes: Annotated[str | None, typer.Option("--notes")] = None,
    biometric_basis: Annotated[
        str | None,
        typer.Option("--biometric-basis", help="Lawful basis for biometric identification."),
    ] = None,
    authorised_by: Annotated[
        str | None, typer.Option("--biometric-authorised-by", help="Who authorised it.")
    ] = None,
    expires: Annotated[
        str | None, typer.Option("--biometric-expires", help="ISO date the authority lapses.")
    ] = None,
    path: Annotated[Path | None, typer.Option("--file", help="Case file location.")] = None,
) -> None:
    """Record case metadata, including any biometric lawful basis."""
    from imgintel.core import casefile

    target = path or casefile.default_path()
    updated = casefile.merge(
        casefile.load(target),
        case_id=case_id,
        operator=operator,
        notes=notes,
        biometric_lawful_basis=biometric_basis,
        biometric_authorised_by=authorised_by,
        biometric_expires=expires,
    )
    written = casefile.save(updated, target)
    console.print(f"[green]Wrote[/green] {written}")
    _show_case(updated)


@case_app.command("show")
def case_show(
    path: Annotated[Path | None, typer.Option("--file")] = None,
) -> None:
    """Show the current case metadata and biometric authorisation state."""
    from imgintel.core import casefile

    target = path or casefile.default_path()
    if not target.exists():
        console.print(f"[dim]No case file at {target}[/dim]")
        raise typer.Exit(0)
    _show_case(casefile.load(target))


def _show_case(case) -> None:
    table = Table(show_header=False, expand=True, box=None)
    table.add_column("k", style="cyan", no_wrap=True)
    table.add_column("v", overflow="fold")
    for label, value in (
        ("Case", case.case_id),
        ("Operator", case.operator),
        ("Notes", case.notes),
        ("Biometric basis", case.biometric_lawful_basis),
        ("Authorised by", case.biometric_authorised_by),
        ("Expires", case.biometric_expires),
    ):
        if value:
            table.add_row(label, escape(str(value)))
    console.print(table)

    if case.biometrics_authorized:
        console.print(
            "[green]Biometric identification authorised.[/green] "
            "Still requires --allow-biometrics on each run."
        )
    elif case.biometric_authorisation_expired:
        console.print(f"[red]Biometric authorisation expired[/red] on {case.biometric_expires}.")
    else:
        console.print("[dim]No biometric authorisation recorded — identification will refuse.[/dim]")


# --------------------------------------------------------------------- db --

db_app = typer.Typer(help="Entity database: enrol and search identities and objects.",
                     no_args_is_help=True)
app.add_typer(db_app, name="db")

DB_DEFAULT = Path("entities.db")


@db_app.command("enrol")
def db_enrol(
    name: Annotated[str, typer.Argument(help="Entity name, e.g. a person or object label.")],
    images: Annotated[list[Path], typer.Argument(help="Reference image(s).")],
    kind: Annotated[
        str, typer.Option("--kind", help="'person' (faces, gated) or 'object'.")
    ] = "object",
    db: Annotated[Path, typer.Option("--db", help="Entity database.")] = DB_DEFAULT,
    notes: Annotated[str | None, typer.Option("--notes")] = None,
    operator: Annotated[str | None, typer.Option("--operator")] = None,
    allow_biometrics: Annotated[
        bool,
        typer.Option("--allow-biometrics", help="Required to enrol faces."),
    ] = False,
    case_file: Annotated[Path | None, typer.Option("--case-file")] = None,
) -> None:
    """Enrol reference images for a person or object."""
    if kind not in ("person", "object"):
        err_console.print("[red]--kind must be 'person' or 'object'[/red]")
        raise typer.Exit(2)

    from imgintel.core import casefile

    case = casefile.load(case_file or casefile.default_path())
    if operator:
        case.operator = operator

    if kind == "person":
        # Enrolling a face creates a biometric template, so it is gated exactly
        # as matching is. Building the database is not a lesser act than
        # searching it.
        refusal = case.biometric_refusal(run_flag=allow_biometrics)
        if refusal:
            err_console.print(
                Panel(
                    f"{escape(refusal)}\n\n"
                    "Enrolling a face creates a biometric template. That is the same "
                    "category of processing as matching one, and is gated the same way.",
                    title="enrolment refused",
                    border_style="red",
                )
            )
            raise typer.Exit(3)

    missing = [p for p in images if not p.exists()]
    if missing:
        err_console.print(f"[red]No such file(s):[/red] {', '.join(str(p) for p in missing)}")
        raise typer.Exit(2)

    count = (
        _enrol_person(name, images, db, notes, case)
        if kind == "person"
        else _enrol_object(name, images, db, notes, case)
    )
    if count:
        console.print(
            f"[green]Enrolled[/green] {count} reference(s) for [bold]{escape(name)}[/bold] "
            f"({kind}) in {db}"
        )
    else:
        err_console.print(
            f"[yellow]No usable references found for {escape(name)}.[/yellow] "
            + (
                "No face was detected, or alignment failed."
                if kind == "person"
                else "The images may lack enough texture for ORB features."
            )
        )
        raise typer.Exit(1)


def _enrol_person(name: str, images: list[Path], db: Path, notes: str | None, case) -> int:
    from imgintel.analyzers.faces import FaceAnalyzer
    from imgintel.core.modelstore import available as models_available
    from imgintel.core.modelstore import status as model_status
    from imgintel.store.entities import EntityStore
    from imgintel.util.embed import FACE_EMBEDDER, embed_faces

    ok, reason = models_available("face", "face-embed")
    if not ok:
        err_console.print(f"[red]{escape(reason)}[/red]")
        raise typer.Exit(1)

    availability = FaceAnalyzer().available()
    if not availability.ok:
        err_console.print(f"[red]{escape(availability.reason)}[/red]")
        raise typer.Exit(1)

    enrolled = 0
    with EntityStore(db) as store:
        entity_id = store.add_entity(name, "person", notes=notes)
        for path in images:
            doc = analyze_image(path, only=["faces"])
            block = doc.analyzers.get("faces")
            if block is None or block.status != "ok":
                continue
            faces = block.data.get("faces", [])
            if not faces:
                console.print(f"[dim]{path.name}: no face detected[/dim]")
                continue
            if len(faces) > 1:
                # Ambiguity at enrolment poisons every later match, so refuse
                # rather than guessing which face is the subject.
                console.print(
                    f"[yellow]{path.name}: {len(faces)} faces — skipped.[/yellow] "
                    "Crop to a single subject before enrolling."
                )
                continue

            ctx_image, scale = _full_image_and_scale(path)
            vectors = embed_faces(ctx_image, faces, model_status("face-embed").path, scale=scale)
            if not vectors or vectors[0] is None:
                console.print(f"[dim]{path.name}: alignment failed[/dim]")
                continue

            sha = doc.analyzers.get("hashes")
            store.add_reference(
                entity_id,
                vectors[0],
                embedder=FACE_EMBEDDER,
                source_path=str(path.resolve()),
                source_sha256=sha.data.get("sha256") if sha else None,
                bbox=faces[0]["bbox"],
                enrolled_by=case.operator,
            )
            enrolled += 1

        store.log(
            "enrol:person",
            f"{name}: {enrolled} reference(s); basis={case.biometric_lawful_basis!r}",
            operator=case.operator,
            case_id=case.case_id,
        )
    return enrolled


def _full_image_and_scale(path: Path):
    """Full-resolution RGB plus the analysis-space scale factor."""
    import numpy as np
    from PIL import Image, ImageOps

    from imgintel.core.context import small_size

    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw) or raw
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    small_w, _ = small_size(width, height)
    return rgb, (width / small_w if small_w else 1.0)


def _enrol_object(name: str, images: list[Path], db: Path, notes: str | None, case) -> int:
    from imgintel.analyzers.objectdb import OBJECT_EMBEDDER, encode_descriptors
    from imgintel.store.entities import EntityStore
    from imgintel.util.orbmatch import MIN_INLIERS, describe

    try:
        import cv2
    except ImportError:
        err_console.print("[red]opencv is not installed[/red] — pip install 'imgintel[logo]'")
        raise typer.Exit(1) from None

    enrolled = 0
    with EntityStore(db) as store:
        entity_id = store.add_entity(name, "object", notes=notes)
        for path in images:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                console.print(f"[dim]{path.name}: could not read[/dim]")
                continue
            keypoints, descriptors = describe(image)
            if descriptors is None or len(keypoints) < MIN_INLIERS:
                console.print(f"[dim]{path.name}: too few features to match reliably[/dim]")
                continue
            store.add_raw_reference(
                entity_id,
                encode_descriptors(keypoints, descriptors, image.shape[:2]),
                embedder=OBJECT_EMBEDDER,
                source_path=str(path.resolve()),
                enrolled_by=case.operator,
            )
            enrolled += 1

        store.log(
            "enrol:object",
            f"{name}: {enrolled} reference(s)",
            operator=case.operator,
            case_id=case.case_id,
        )
    return enrolled


@db_app.command("list")
def db_list(
    db: Annotated[Path, typer.Option("--db")] = DB_DEFAULT,
    kind: Annotated[str | None, typer.Option("--kind")] = None,
) -> None:
    """List enrolled entities."""
    from imgintel.store.entities import EntityStore

    if not db.exists():
        err_console.print(f"[yellow]No entity database at[/yellow] {db}")
        raise typer.Exit(1)

    with EntityStore(db) as store:
        entities = store.entities(kind)  # type: ignore[arg-type]
        stats = store.stats()

    table = Table(header_style="bold", expand=True)
    table.add_column("Name", style="cyan")
    table.add_column("Kind", no_wrap=True)
    table.add_column("Refs", justify="right", no_wrap=True)
    table.add_column("Enrolled", no_wrap=True)
    table.add_column("Notes", overflow="fold")
    for entity in entities:
        style = "yellow" if entity.kind == "person" else ""
        table.add_row(
            entity.name,
            Text(entity.kind, style=style),
            str(entity.reference_count),
            entity.created_utc[:10],
            escape(entity.notes or ""),
        )
    console.print(table)
    console.print(
        f"[dim]{stats['people']} person, {stats['objects']} object · "
        f"{stats['refs']} references · {db}[/dim]"
    )


@db_app.command("forget")
def db_forget(
    name: Annotated[str, typer.Argument(help="Entity to delete.")],
    db: Annotated[Path, typer.Option("--db")] = DB_DEFAULT,
    kind: Annotated[str | None, typer.Option("--kind")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
) -> None:
    """Delete an entity and all its references."""
    from imgintel.store.entities import EntityStore

    if not db.exists():
        err_console.print(f"[yellow]No entity database at[/yellow] {db}")
        raise typer.Exit(1)
    if not yes and not typer.confirm(f"Permanently delete '{name}' and its references?"):
        raise typer.Exit(0)

    with EntityStore(db) as store:
        removed = store.forget(name, kind)  # type: ignore[arg-type]
        store.log("forget", f"{name} ({kind or 'any kind'})")

    if removed:
        console.print(f"[green]Deleted[/green] {removed} entit(y/ies) named {escape(name)}")
    else:
        err_console.print(f"[yellow]No entity named[/yellow] {escape(name)}")
        raise typer.Exit(1)


@db_app.command("audit")
def db_audit(
    db: Annotated[Path, typer.Option("--db")] = DB_DEFAULT,
    limit: Annotated[int, typer.Option("--limit")] = 50,
) -> None:
    """Show the entity database audit trail."""
    from imgintel.store.entities import EntityStore

    if not db.exists():
        err_console.print(f"[yellow]No entity database at[/yellow] {db}")
        raise typer.Exit(1)

    with EntityStore(db) as store:
        entries = store.audit_log(limit)

    table = Table(header_style="bold", expand=True)
    table.add_column("When", no_wrap=True)
    table.add_column("Action", style="cyan", no_wrap=True)
    table.add_column("Operator", no_wrap=True)
    table.add_column("Case", no_wrap=True)
    table.add_column("Detail", overflow="fold")
    for entry in entries:
        table.add_row(
            entry["at_utc"], entry["action"], entry["operator"] or "—",
            entry["case_id"] or "—", escape(entry["detail"] or ""),
        )
    console.print(table)


# ----------------------------------------------------------------- models --

models_app = typer.Typer(help="Manage model weights.", no_args_is_help=True)
app.add_typer(models_app, name="models")


@models_app.command("list")
def models_list(
    verify: Annotated[bool, typer.Option("--verify", help="Re-hash every present model.")] = False,
) -> None:
    """Show every known model, whether it is present, and its licence."""
    from imgintel.core.modelstore import all_status, cache_dir, load_receipts

    receipts = load_receipts()
    table = Table(header_style="bold", expand=True)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Kind", no_wrap=True)
    table.add_column("Licence", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Detail", overflow="fold")

    unverified: list[str] = []
    for st in all_status(verify=verify):
        if not st.present:
            status_text = Text("missing", style="yellow")
            detail = st.note
        elif st.verified is False:
            status_text = Text("CORRUPT", style="bold red")
            detail = st.note
        else:
            status_text = Text("ready", style="green")
            size = f"{(st.size_bytes or 0) / 1e6:.1f} MB"
            detail = f"{st.description} · {size}"
            if st.sha256:
                detail += f" · {st.sha256[:16]}…"
            if st.note:
                detail += f"\n{st.note}"
            # How it arrived is part of what the model is.
            if receipts.get(st.key, {}).get("tls_verified") is False:
                unverified.append(st.key)
                detail += "\nfetched with TLS verification DISABLED"
        table.add_row(st.key, st.kind, st.licence, status_text, escape(detail))

    console.print(table)
    if unverified:
        console.print(
            f"[red]{', '.join(unverified)} were downloaded without certificate "
            "verification.[/red] Re-pull them securely, or verify against the upstream "
            "checksum and pin."
        )
    console.print(f"[dim]Download cache: {cache_dir()}[/dim]")


@models_app.command("pull")
def models_pull(
    keys: Annotated[
        list[str] | None, typer.Argument(help="Model keys to fetch. Omit for all.")
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Re-download even if present.")] = False,
    insecure: Annotated[
        bool,
        typer.Option(
            "--insecure",
            help="Skip TLS certificate verification. Last resort — see the note it prints.",
        ),
    ] = False,
) -> None:
    """Download model weights into the local cache."""
    from imgintel.core.modelstore import MODELS, CertificateError, DownloadError, pull

    targets = keys or list(MODELS)
    unknown = [k for k in targets if k not in MODELS]
    if unknown:
        err_console.print(f"[red]Unknown model(s):[/red] {', '.join(unknown)}")
        raise typer.Exit(2)

    if insecure:
        unpinned = [k for k in targets if not MODELS[k].bundled and not MODELS[k].sha256]
        console.print(
            Panel(
                "TLS certificate verification is [bold]disabled[/bold] for this download.\n"
                + (
                    f"[yellow]{', '.join(unpinned)} are unpinned[/yellow], so there is no hash "
                    "check either — this download has [bold]no integrity guarantee at all[/bold], "
                    "and ONNX graphs are parsed by onnxruntime.\n"
                    if unpinned
                    else ""
                )
                + "Verify the file against the upstream checksum afterwards and run "
                "[bold]imgintel models pin[/bold].\n"
                "This is recorded in the model receipt and appears in evidence provenance.",
                title="insecure download",
                border_style="red",
            )
        )

    failures = 0
    for key in targets:
        spec = MODELS[key]
        if spec.bundled:
            console.print(
                f"[dim]{key}: ships with {spec.provided_by}, nothing to download[/dim]"
            )
            continue
        try:
            with console.status(f"Fetching {key}…"):
                st = pull(key, force=force, insecure=insecure)
            console.print(f"[green]{key}[/green] ready · {st.sha256[:16] if st.sha256 else ''}…")
            if st.note:
                console.print(f"  [yellow]{escape(st.note)}[/yellow]")
        except CertificateError as exc:
            failures += 1
            err_console.print(
                Panel(escape(str(exc)), title=f"{key}: certificate verification failed",
                      border_style="red")
            )
        except DownloadError as exc:
            failures += 1
            err_console.print(f"[red]{key} failed:[/red] {escape(str(exc))}")

    raise typer.Exit(1 if failures else 0)


@models_app.command("import")
def models_import(
    source: Annotated[
        Path, typer.Argument(help="A downloaded model file, or a folder of them.")
    ],
    key: Annotated[
        str | None, typer.Option("--key", help="Model key, if the filename does not identify it.")
    ] = None,
    move: Annotated[bool, typer.Option("--move", help="Move rather than copy.")] = False,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an installed model.")] = False,
) -> None:
    """Install model weights you downloaded yourself.

    The alternative to `pull` when the machine cannot reach the mirrors.
    """
    from imgintel.core.modelstore import DownloadError, import_directory, import_file

    if not source.exists():
        err_console.print(f"[red]Error:[/red] no such path: {source}")
        raise typer.Exit(2)

    try:
        results = (
            import_directory(source, force=force)
            if source.is_dir()
            else [import_file(source, key, move=move, force=force)]
        )
    except DownloadError as exc:
        err_console.print(f"[red]Import failed:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    if not results:
        err_console.print(f"[yellow]Nothing recognisable in[/yellow] {source}")
        raise typer.Exit(1)

    for st in results:
        console.print(f"[green]{st.key}[/green] installed → {st.path}")
        if st.note:
            console.print(f"  [yellow]{escape(st.note)}[/yellow]")


@models_app.command("pin")
def models_pin(
    keys: Annotated[
        list[str] | None, typer.Argument(help="Model keys to pin. Omit for all installed.")
    ] = None,
) -> None:
    """Record the digest of an installed model so later checks are exact.

    Verify the file against the upstream checksum first — pinning a tampered
    model only makes the tampering reproducible.
    """
    from imgintel.core.modelstore import DownloadError, all_status
    from imgintel.core.modelstore import pin as pin_model

    targets = keys or [s.key for s in all_status() if s.present and not s.sha256]
    if not targets:
        console.print("[dim]Nothing to pin — every installed model already has a digest.[/dim]")
        return

    for key in targets:
        try:
            st = pin_model(key)
        except DownloadError as exc:
            err_console.print(f"[red]{key}:[/red] {exc}")
            continue
        console.print(f"[green]{key}[/green] pinned at {st.sha256}")


@models_app.command("verify")
def models_verify() -> None:
    """Re-hash every present model against its pinned digest."""
    from imgintel.core.modelstore import all_status

    checked = all_status(verify=True)
    bad = [s for s in checked if s.present and s.verified is False]
    present = [s for s in checked if s.present]
    unpinned = [s for s in present if s.verified is None and not s.kind == "bundled"]

    if bad:
        for st in bad:
            err_console.print(f"[red]{st.key}[/red] hash mismatch — {st.path}")
        raise typer.Exit(1)

    console.print(f"[green]{len(present)} model(s) present, no hash mismatches.[/green]")
    if unpinned:
        console.print(
            f"[yellow]{len(unpinned)} unpinned:[/yellow] "
            f"{', '.join(s.key for s in unpinned)} — run [bold]imgintel models pin[/bold] "
            "after verifying them against the upstream checksum."
        )


@models_app.command("where")
def models_where() -> None:
    """Show every directory searched for models."""
    from imgintel.core.modelstore import MODEL_DIR_ENV, search_dirs

    for directory in search_dirs():
        marker = "[green]exists[/green]" if directory.is_dir() else "[dim]absent[/dim]"
        console.print(f"  {marker}  {directory}")
    console.print(
        f"[dim]Add more with {MODEL_DIR_ENV} "
        f"(separate with '{__import__('os').pathsep}').[/dim]"
    )


# ----------------------------------------------------------------- doctor --


@app.command()
def doctor() -> None:
    """Check the environment and report what is missing and how to fix it."""
    from imgintel.core.deps import check_all

    console.print(Panel(f"imgintel {__version__}  ·  schema {SCHEMA_VERSION}", expand=False))

    registry = Registry.discover()
    _warn_load_errors(registry)

    table = Table(title="Analyzers", header_style="bold", expand=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Cost", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Detail", overflow="fold")

    unavailable = 0
    for analyzer in registry.all():
        avail = analyzer.available()
        if avail.ok:
            status, detail = Text("ready", style="green"), analyzer.description
        else:
            unavailable += 1
            status = Text("unavailable", style="yellow")
            detail = f"{avail.reason} — {avail.hint}" if avail.hint else avail.reason
        table.add_row(analyzer.name, analyzer.cost.value, status, escape(detail))
    console.print(table)

    deps = Table(title="Optional dependencies", header_style="bold", expand=True)
    deps.add_column("Dependency", style="cyan", no_wrap=True)
    deps.add_column("Kind", no_wrap=True)
    deps.add_column("Status", no_wrap=True)
    deps.add_column("Enables / fix", overflow="fold")

    missing = 0
    for dep in check_all():
        if dep.present:
            status = Text(dep.version or "present", style="green")
            detail = dep.enables
        else:
            missing += 1
            status = Text("missing", style="yellow")
            # escape(): hints contain pip extras like imgintel[exif], which
            # rich would otherwise swallow as markup.
            detail = f"{escape(dep.enables)}\n[dim]fix:[/dim] {escape(dep.hint)}"
        deps.add_row(dep.key, dep.kind, status, detail)
    console.print(deps)

    from imgintel.core.modelstore import all_status

    models = Table(title="Models", header_style="bold", expand=True)
    models.add_column("Model", style="cyan", no_wrap=True)
    models.add_column("Licence", no_wrap=True)
    models.add_column("Status", no_wrap=True)
    models.add_column("Detail", overflow="fold")
    absent_models = 0
    for st in all_status():
        if st.present:
            models.add_row(st.key, st.licence, Text("ready", style="green"), escape(st.description))
        else:
            absent_models += 1
            models.add_row(st.key, st.licence, Text("missing", style="yellow"), escape(st.note))
    console.print(models)

    if missing or unavailable or absent_models:
        console.print(
            f"[yellow]{unavailable} analyzer(s) unavailable, "
            f"{missing} optional dependency and {absent_models} model(s) missing.[/yellow] "
            "Core analysis still works without them."
        )
    else:
        console.print("[green]Everything is installed.[/green]")


# ---------------------------------------------------------------- plugins --


plugins_app = typer.Typer(
    help="Inspect, scaffold and validate analyzers.", no_args_is_help=False, invoke_without_command=True
)
app.add_typer(plugins_app, name="plugins")


@plugins_app.callback(invoke_without_command=True)
def plugins_main(
    ctx: typer.Context,
    allow_local: Annotated[
        bool, typer.Option("--allow-local", help="Include local .py plugins.")
    ] = False,
) -> None:
    """List every registered analyzer and where it came from."""
    if ctx.invoked_subcommand is not None:
        return

    registry = Registry.discover(allow_local=allow_local)
    _warn_load_errors(registry)

    table = Table(header_style="bold", expand=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Ver", no_wrap=True)
    table.add_column("Cost", no_wrap=True)
    table.add_column("Requires", no_wrap=True)
    table.add_column("Origin", no_wrap=True)
    table.add_column("Description", overflow="fold")

    for a in registry.all():
        origin = registry.origin(a.name)
        style = "green" if origin.startswith("package:") else (
            "yellow" if origin.startswith(("rules:", "local:")) else ""
        )
        table.add_row(
            a.name,
            a.version,
            a.cost.value,
            ", ".join(a.requires) or "—",
            Text(origin, style=style),
            escape(a.description),
        )
    console.print(table)
    console.print(
        f"[dim]{len(registry)} analyzers · plugin API {PLUGIN_API_VERSION}[/dim]"
    )


@plugins_app.command("dirs")
def plugins_dirs() -> None:
    """Show where plugins and rule files are loaded from."""
    from imgintel.core.registry import user_rules_dir

    table = Table(show_header=False, expand=True, box=None)
    table.add_column("k", style="cyan", no_wrap=True)
    table.add_column("v", overflow="fold")
    table.add_row("Rule files", f"{user_rules_dir()}  [dim](loaded automatically)[/dim]")
    table.add_row(
        "Python plugins",
        f"{user_plugin_dir()}  [dim](needs --allow-local-plugins)[/dim]",
    )
    table.add_row("Installed packages", "any package with an 'imgintel.analyzers' entry point")
    console.print(table)


@plugins_app.command("new")
def plugins_new(
    name: Annotated[str, typer.Argument(help="Analyzer name, e.g. 'anpr'.")],
    kind: Annotated[
        str, typer.Option("--kind", help="'python' for a package, 'rules' for a YAML ruleset.")
    ] = "python",
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Where to write it.")
    ] = None,
) -> None:
    """Scaffold a working plugin, ready to install and edit."""
    from imgintel.core.registry import user_rules_dir
    from imgintel.plugins.scaffold import ScaffoldError, scaffold_python, scaffold_rules

    if kind not in ("python", "rules"):
        err_console.print("[red]--kind must be 'python' or 'rules'[/red]")
        raise typer.Exit(2)

    try:
        if kind == "rules":
            written = scaffold_rules(out or user_rules_dir(), name)
            follow_up = (
                "It is already active — rule files load automatically.\n"
                f"Check it with: [bold]imgintel plugins validate {written[0]}[/bold]"
            )
        else:
            written = scaffold_python(out or Path.cwd(), name)
            package = written[0].parent.parent.parent
            follow_up = (
                f"cd {package}\n"
                "pip install -e .\n"
                f"imgintel plugins        [dim]# {name} should now be listed[/dim]"
            )
    except ScaffoldError as exc:
        err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(2) from exc

    table = Table(show_header=False, expand=True, box=None)
    table.add_column("f", style="cyan")
    for path in written:
        table.add_row(str(path))
    console.print(Panel(table, title=f"created {kind} plugin '{name}'", border_style="green"))
    console.print(follow_up)


@plugins_app.command("validate")
def plugins_validate(
    target: Annotated[
        str | None,
        typer.Argument(help="Analyzer name, or a path to a rule file. Omit to check all."),
    ] = None,
    sample: Annotated[
        Path | None,
        typer.Option("--sample", help="Image to exercise the analyzer against."),
    ] = None,
    allow_local: Annotated[bool, typer.Option("--allow-local")] = False,
) -> None:
    """Check analyzers against the plugin contract.

    With --sample it also runs them, which is the only way to catch a CHEAP
    analyzer that decodes pixels or one that is not deterministic.
    """
    from imgintel.plugins.validate import validate, validate_ruleset

    reports = []
    if target and Path(target).exists() and Path(target).is_file():
        reports.append(validate_ruleset(Path(target)))
    else:
        registry = Registry.discover(allow_local=allow_local)
        _warn_load_errors(registry)
        if target:
            analyzer = registry.get(target)
            if analyzer is None:
                err_console.print(
                    f"[red]Unknown analyzer[/red] {target!r}. "
                    f"Available: {', '.join(registry.names())}"
                )
                raise typer.Exit(2)
            reports.append(validate(analyzer, sample))
        else:
            reports = [validate(a, sample) for a in registry.all()]

    failed = 0
    for report in reports:
        if report.ok and not report.warnings:
            console.print(f"[green]ok[/green]      {report.analyzer}")
            continue
        if not report.ok:
            failed += 1
        style = "red" if not report.ok else "yellow"
        label = "FAIL" if not report.ok else "warn"
        console.print(f"[{style}]{label}[/{style}]    {report.analyzer}")
        for issue in report.issues:
            marker = "[red]•[/red]" if issue.level == "error" else "[yellow]•[/yellow]"
            console.print(f"        {marker} {escape(issue.message)}")

    total = len(reports)
    if failed:
        console.print(f"\n[red]{failed} of {total} failed.[/red]")
        raise typer.Exit(1)
    console.print(f"\n[green]{total} analyzer(s) conform.[/green]")
    if sample is None:
        console.print(
            "[dim]Pass --sample IMAGE to also check cost honesty and determinism.[/dim]"
        )


@app.command()
def profiles() -> None:
    """Show the execution profiles and which analyzers each one runs."""
    registry = Registry.discover()
    table = Table(header_style="bold", expand=True)
    table.add_column("Profile", style="cyan", no_wrap=True)
    table.add_column("Description", overflow="fold")
    table.add_column("Analyzers", overflow="fold")

    for name, prof in PROFILES.items():
        chosen = plan(registry, profile=name)
        label = f"{name} [dim](default)[/dim]" if name == DEFAULT_PROFILE else name
        table.add_row(label, prof.description, ", ".join(a.name for a in chosen) or "—")
    console.print(table)


@app.command(name="schema")
def dump_schema(
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Write to a file.")] = None,
) -> None:
    """Print the JSON Schema of the findings document."""
    text = json.dumps(FindingsDocument.model_json_schema(), indent=2)
    if out:
        out.write_text(text, encoding="utf-8")
        console.print(f"[green]Wrote[/green] {out}")
    else:
        sys.stdout.write(text + "\n")


# --------------------------------------------------------------- rendering --


def _split(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _warn_load_errors(registry: Registry) -> None:
    for error in registry.errors:
        err_console.print(f"[yellow]Plugin load failed[/yellow] {error.source}: {error.error}")


def _render(doc: FindingsDocument, *, show_data: bool = False) -> None:
    target = doc.target
    header = Text()
    header.append(target.filename, style="bold")
    header.append(f"\n{target.path}\n", style="dim")
    header.append(
        f"{doc.summary.findings_total} findings  ·  "
        f"{doc.summary.analyzers_ok} ok  ·  "
        f"{doc.summary.analyzers_unavailable} unavailable  ·  "
        f"{doc.summary.analyzers_failed} failed  ·  "
        f"{doc.run.duration_ms:.0f} ms"
    )
    console.print(Panel(header, title="imgintel", border_style="blue"))

    if doc.findings:
        table = Table(header_style="bold", expand=True, show_lines=False)
        table.add_column("Sev", no_wrap=True, width=7)
        table.add_column("Category", no_wrap=True, style="magenta")
        table.add_column("Finding", overflow="fold")
        table.add_column("Value", overflow="fold")

        for f in doc.findings:
            style = SEVERITY_STYLE[f.severity]
            value = "" if f.value is None else str(f.value)
            if len(value) > 90:
                value = value[:87] + "..."
            label = f.label
            if f.detail:
                label = f"{label}\n[dim]{f.detail}[/dim]"
            table.add_row(Text(f.severity, style=style), f.category, label, value)
        console.print(table)
    else:
        console.print("[dim]No findings.[/dim]")

    # Analyzer status — only the ones that need attention, unless verbose.
    problems = [b for b in doc.analyzers.values() if b.status != "ok"]
    if problems:
        table = Table(title="Analyzers not run", header_style="bold", expand=True)
        table.add_column("Analyzer", style="cyan", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column("Reason", overflow="fold")
        for b in sorted(problems, key=lambda x: x.analyzer):
            table.add_row(
                b.analyzer, Text(b.status, style=STATUS_STYLE.get(b.status, "")), b.error or ""
            )
        console.print(table)

    if show_data:
        for name, block in sorted(doc.analyzers.items()):
            if block.status != "ok":
                continue
            body = json.dumps(block.data, indent=2, default=str)
            if len(body) > 4000:
                body = body[:4000] + "\n... (truncated; use --json for the full document)"
            console.print(Panel(body, title=f"{name} data", border_style="dim"))


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
