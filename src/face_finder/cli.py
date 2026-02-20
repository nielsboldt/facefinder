"""Command-line interface for Face Finder."""

import sys
from pathlib import Path

import click

from .detector import (
    IMAGE_EXTENSIONS,
    analyze_reference_images,
    extract_face_thumbnail,
    load_encoding,
    load_reference_encodings,
    save_encoding,
)
from .matcher import process_images, process_images_prefilter_only
from .prefilter import PrefilterResult


def iter_paths_from_stdin():
    """Read image paths from stdin, one per line.

    Yields Path objects for valid image files. Logs warnings to stderr
    for non-existent paths.
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        path = Path(line)
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.exists():
                yield path
            else:
                click.echo(f"Warning: file not found: {path}", err=True)


@click.group()
@click.version_option()
def main() -> None:
    """Face Finder - Find images containing a specific person."""
    pass


@main.command()
@click.option(
    "--reference-dir",
    "-r",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing reference images to analyze.",
)
@click.option(
    "--tolerance",
    "-t",
    default=0.40,
    type=float,
    help="Match tolerance (0.3=very strict, 0.40=default, 0.6=permissive). Lower values reduce false positives.",
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(["hog", "cnn", "auto"]),
    default="hog",
    help="Face detection model: 'hog' (fast), 'cnn' (slow, handles profiles), 'auto' (hog first, cnn fallback).",
)
@click.option(
    "--save-encoding",
    "-e",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Save the face encoding to a .npy file for later use.",
)
@click.option(
    "--save-thumbnail",
    "-T",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Save a thumbnail of the detected face for verification.",
)
def analyze(
    reference_dir: Path,
    tolerance: float,
    model: str,
    save_encoding: Path | None,
    save_thumbnail: Path | None,
) -> None:
    """Analyze reference images and identify the common face."""
    click.echo(f"Analyzing reference images in: {reference_dir}")
    click.echo(f"Tolerance: {tolerance}")
    model_desc = {
        "hog": "hog (fast, frontal faces)",
        "cnn": "cnn (slow, handles profiles)",
        "auto": "auto (hog first, cnn fallback if no faces found)",
    }
    click.echo(f"Model: {model_desc.get(model, model)}")
    if model in ("cnn", "auto"):
        click.echo("  Note: CNN fallback is slow without GPU (~30s per image)")
    click.echo("")

    def on_progress(image_path, faces_found, model_used, prefilter_info=None):
        suffix = f" [via {model_used}]" if model == "auto" else ""
        # Include prefilter scores when available (CNN/auto mode)
        pf_str = ""
        if prefilter_info:
            pf_parts = []
            if prefilter_info.entropy is not None:
                pf_parts.append(f"entropy={prefilter_info.entropy:.1f}")
            if prefilter_info.skin_percentage is not None:
                pf_parts.append(f"skin={prefilter_info.skin_percentage:.1f}%")
            if pf_parts:
                pf_str = f" [{', '.join(pf_parts)}]"
        click.echo(f"  Processing: {image_path.name} -> {faces_found} face(s){suffix}{pf_str}")

    result = analyze_reference_images(reference_dir, tolerance, model=model, progress_callback=on_progress)
    click.echo("")

    if result is None:
        click.echo("Error: No faces found in reference images.", err=True)
        raise SystemExit(1)

    # Summary
    click.echo("=" * 50)
    click.echo("Analysis Results")
    click.echo("=" * 50)
    click.echo(f"Total images analyzed: {result.total_images}")
    click.echo(f"Total faces detected:  {result.total_faces_detected}")
    click.echo(f"Common face found in:  {result.match_count}/{result.total_images} images")
    click.echo("")

    # Per-image details
    click.echo("Images with common face:")
    for img in result.images_with_face:
        faces = result.faces_per_image.get(img, 0)
        distance = result.distances_to_common.get(img)
        dist_str = f", distance={distance:.2f}" if distance is not None else ""
        click.echo(f"  [+] {img.name} ({faces} face{'s' if faces != 1 else ''}{dist_str})")

    if result.images_without_face:
        click.echo("")
        click.echo("Images WITHOUT common face:")
        for img in result.images_without_face:
            faces = result.faces_per_image.get(img, 0)
            distance = result.distances_to_common.get(img)
            if faces == 0:
                click.echo(f"  [-] {img.name} (no faces detected)")
            else:
                dist_str = f", distance={distance:.2f}" if distance is not None else ""
                click.echo(f"  [-] {img.name} ({faces} face{'s' if faces != 1 else ''}{dist_str}, different person)")

    # Display skin color information
    if result.skin_color:
        click.echo("")
        click.echo("Skin color extracted:")
        r, g, b = result.skin_color["median_rgb"]
        click.echo(f"  Median RGB: ({r}, {g}, {b})")
        sr, sg, sb = result.skin_color["std_rgb"]
        click.echo(f"  Tolerance: +/- ({sr:.0f}, {sg:.0f}, {sb:.0f})")
        click.echo(f"  Sampled from: {result.skin_color['image_count']} images, {result.skin_color['sample_count']} pixels")
    else:
        click.echo("")
        click.echo("Skin color: not extracted (insufficient skin pixels in face regions)")

    # Save encoding if requested
    if save_encoding:
        from .detector import save_encoding as do_save_encoding
        do_save_encoding(result.encoding, save_encoding, skin_color=result.skin_color)
        click.echo("")
        click.echo(f"Encoding saved to: {save_encoding}")

    # Save thumbnail if requested
    if save_thumbnail and result.images_with_face:
        # Use first matching image to extract thumbnail
        source_image = result.images_with_face[0]
        success = extract_face_thumbnail(
            source_image,
            result.encoding,
            save_thumbnail,
            tolerance,
            model=model,
        )
        click.echo("")
        if success:
            click.echo(f"Thumbnail saved to: {save_thumbnail}")
        else:
            click.echo(f"Warning: Could not extract thumbnail from {source_image.name}")


@main.command()
@click.option(
    "--reference-dir",
    "-r",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing reference images of the person to find.",
)
@click.option(
    "--encoding",
    "-e",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Pre-computed face encoding file (.npy) from 'analyze' command.",
)
@click.option(
    "--search-dir",
    "-s",
    required=False,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory to search for images. If omitted, reads image paths from stdin.",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to copy matching images to.",
)
@click.option(
    "--tolerance",
    "-t",
    default=0.40,
    type=float,
    help="Match tolerance (0.3=very strict, 0.40=default, 0.6=permissive). Lower values reduce false positives.",
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(["hog", "cnn", "auto"]),
    default="hog",
    help="Face detection model: 'hog' (fast), 'cnn' (slow, handles profiles), 'auto' (hog first, cnn fallback).",
)
@click.option(
    "--recursive",
    "-R",
    is_flag=True,
    default=False,
    help="Search subdirectories recursively.",
)
@click.option(
    "--limit",
    "-n",
    type=int,
    default=None,
    help="Maximum number of matches to find (stops early when reached).",
)
@click.option(
    "--log",
    "-l",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write detailed processing log to file.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show distance values for each image (useful for tuning tolerance).",
)
def find(
    reference_dir: Path | None,
    encoding: Path | None,
    search_dir: Path | None,
    output_dir: Path,
    tolerance: float,
    model: str,
    recursive: bool,
    limit: int | None,
    log: Path | None,
    verbose: bool,
) -> None:
    """Find images containing a specific person.

    Input modes:
    - With --search-dir: scans the directory for images
    - Without --search-dir: reads image paths from stdin (one per line)

    Output modes:
    - Non-verbose (default): prints every processed image path to stdout (enables resumability)
    - Verbose (-v): prints detailed status for each image
    - With --log: writes detailed log to file regardless of verbose mode

    Matching images are copied to --output-dir regardless of output mode.
    """
    from typing import Iterator

    # Validate that exactly one of reference_dir or encoding is provided
    if reference_dir is None and encoding is None:
        click.echo("Error: Must provide either --reference-dir or --encoding.", err=True)
        raise SystemExit(1)
    if reference_dir is not None and encoding is not None:
        click.echo("Error: Cannot provide both --reference-dir and --encoding.", err=True)
        raise SystemExit(1)

    # Helper to write to debug log file
    log_file = None
    if log:
        log_file = open(log, "w")

    def log_write(msg: str) -> None:
        """Write message to debug log file if open."""
        if log_file:
            log_file.write(msg + "\n")
            log_file.flush()

    def info_output(msg: str) -> None:
        """Write informational message to appropriate destination."""
        if verbose:
            click.echo(msg)
            sys.stdout.flush()
        log_write(msg)

    # Initialize debug log header
    if log:
        log_write("Face Finder Log")
        log_write(f"Search dir: {search_dir if search_dir else 'stdin'}")
        log_write(f"Model: {model}")
        log_write("=" * 50)

    # Load encoding from file or analyze reference images
    skin_info: dict | None = None
    if encoding is not None:
        info_output(f"Loading encoding from: {encoding}")
        reference_encoding, skin_info = load_encoding(encoding)
        info_output("Encoding loaded successfully.")
    else:
        info_output(f"Loading reference images from: {reference_dir}")
        info_output(f"Model: {model}" + (" (better for profiles/side views)" if model == "cnn" else " (fast, frontal faces)"))
        result = load_reference_encodings(reference_dir, tolerance, model=model)

        if result is None:
            click.echo("Error: No faces found in reference images.", err=True)
            raise SystemExit(1)

        reference_encoding, images_with_face, total_images = result
        info_output(f"Found common face appearing in {images_with_face}/{total_images} reference images.")
        if images_with_face < total_images:
            info_output(
                f"  Note: Face not found in {total_images - images_with_face} image(s). "
                "These may have no detectable faces or contain different people."
            )

    # Display targeted prefilter status
    if skin_info and model in ("cnn", "auto"):
        r, g, b = skin_info["median_rgb"]
        info_output(f"Targeted skin prefilter: enabled (RGB: {r}, {g}, {b})")
    elif model in ("cnn", "auto"):
        info_output("Targeted skin prefilter: disabled (using generic heuristic)")

    # Determine input source
    using_stdin = search_dir is None
    if using_stdin:
        info_output("Reading image paths from stdin...")
        image_source: Path | Iterator[Path] = iter_paths_from_stdin()
    else:
        info_output(f"Scanning images in: {search_dir}")
        if recursive:
            info_output("(recursive mode enabled)")
        image_source = search_dir

    info_output("")

    # Track progress with per-image logging
    scanned = 0
    matches_found = 0
    errors_found = 0

    def log_before_process(image_path: Path, model_used: str) -> None:
        """Log BEFORE processing each image (critical for crash diagnosis)."""
        msg = f"[PROCESSING] {image_path} model={model_used}"
        if verbose:
            click.echo(msg)
            sys.stdout.flush()
        log_write(msg)

    def format_prefilter_scores(match_info) -> str:
        """Format prefilter scores for display."""
        parts = []
        if match_info.prefilter_entropy is not None:
            parts.append(f"entropy={match_info.prefilter_entropy:.1f}")
        if match_info.prefilter_ref_skin_pct is not None:
            parts.append(f"ref_skin={match_info.prefilter_ref_skin_pct:.1f}%")
        elif match_info.prefilter_skin_pct is not None:
            parts.append(f"skin={match_info.prefilter_skin_pct:.1f}%")
        return ", ".join(parts)

    def log_prefilter(image_path: Path, prefilter: PrefilterResult) -> None:
        """Log prefilter result immediately after it's computed."""
        scores = []
        if prefilter.entropy is not None:
            scores.append(f"entropy={prefilter.entropy:.1f}")
        if prefilter.reference_skin_percentage is not None:
            scores.append(f"ref_skin={prefilter.reference_skin_percentage:.1f}%")
        elif prefilter.skin_percentage is not None:
            scores.append(f"skin={prefilter.skin_percentage:.1f}%")
        if prefilter.blob_aspect_ratio is not None:
            scores.append(f"aspect={prefilter.blob_aspect_ratio:.2f}")

        if prefilter.should_skip:
            status = f"SKIP ({prefilter.reason})"
        else:
            status = "PASS"

        msg = f"  -> prefilter: {', '.join(scores)} -> {status}"
        if verbose:
            click.echo(msg)
            sys.stdout.flush()
        log_write(msg)

    def update_progress(image_path: Path, matched: bool, error: bool = False, error_msg: str = None, match_info=None) -> None:
        nonlocal scanned, matches_found, errors_found
        scanned += 1
        if error:
            errors_found += 1
            status = f"ERROR: {error_msg}" if error_msg else "ERROR"
        elif matched:
            matches_found += 1
            if match_info:
                scores = format_prefilter_scores(match_info)
                if scores:
                    status = f"MATCH (distance={match_info.min_distance:.2f}, faces={match_info.num_faces}, {scores})"
                else:
                    status = f"MATCH (distance={match_info.min_distance:.2f}, faces={match_info.num_faces})"
            else:
                status = "MATCH"
        else:
            if match_info:
                scores = format_prefilter_scores(match_info)
                if match_info.prefilter_skipped:
                    if scores:
                        status = f"no match (skipped: {match_info.prefilter_reason}, {scores})"
                    else:
                        status = f"no match (skipped: {match_info.prefilter_reason})"
                elif match_info.num_faces > 0:
                    if scores:
                        status = f"no match (distance={match_info.min_distance:.2f}, faces={match_info.num_faces}, {scores})"
                    else:
                        status = f"no match (distance={match_info.min_distance:.2f}, faces={match_info.num_faces})"
                else:
                    if scores:
                        status = f"no match (no faces detected, {scores})"
                    else:
                        status = "no match (no faces detected)"
            else:
                status = "no match"

        # Detailed output to verbose stdout or log file
        detail_msg = f"  [{scanned}] {image_path.name} - {status}"
        if verbose:
            click.echo(detail_msg)
            sys.stdout.flush()
        log_write(detail_msg)

        # In non-verbose mode, output every processed image path to stdout
        # (enables resumability - can diff output against input to find unprocessed images)
        if not verbose:
            click.echo(str(image_path))
            sys.stdout.flush()

    # Process images
    info_output("Starting image processing...")
    interrupted = False
    try:
        result = process_images(
            image_source=image_source,
            output_dir=output_dir,
            reference_encoding=reference_encoding,
            tolerance=tolerance,
            recursive=recursive,
            limit=limit,
            progress_callback=update_progress,
            pre_process_callback=log_before_process if verbose or log else None,
            model=model,
            verbose=verbose or (log is not None),
            prefilter_callback=log_prefilter if (verbose or log) else None,
            skin_info=skin_info,
        )
    except KeyboardInterrupt:
        interrupted = True
        # Create partial result from tracked progress
        from .matcher import MatchResult
        result = MatchResult()
        result.total_scanned = scanned
        result.matches_found = matches_found
        result.errors = errors_found

    # Build summary
    summary_lines = [
        "",
        "=" * 50,
        "Summary",
        "=" * 50,
        f"  Images scanned: {result.total_scanned}",
        f"  Matches found:  {result.matches_found}",
        f"  Errors:         {result.errors}",
        f"  Output:         {output_dir}",
    ]

    if result.total_scanned == 0:
        summary_lines.append("")
        source_desc = "stdin" if using_stdin else "the search directory"
        summary_lines.append(f"  No images found in {source_desc}.")

    if interrupted:
        summary_lines.append("  (interrupted by user)")
    elif limit is not None and result.matches_found >= limit:
        summary_lines.append(f"  (stopped early after reaching limit of {limit})")

    # Output summary to appropriate destination
    # In non-verbose mode: summary goes to stderr (to keep stdout clean for piping)
    # In verbose mode: summary goes to stdout
    # Always write to debug log file if provided
    for line in summary_lines:
        if verbose:
            click.echo(line)
        else:
            click.echo(line, err=True)
        log_write(line)

    # Close debug log file
    if log_file:
        log_file.close()
        if verbose:
            click.echo(f"  Log file: {log}")
        else:
            click.echo(f"  Log file: {log}", err=True)

    sys.stdout.flush()


@main.command()
@click.option(
    "--search-dir",
    "-s",
    required=False,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory to search for images. If omitted, reads image paths from stdin.",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to copy passing images to.",
)
@click.option(
    "--encoding",
    "-e",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional: encoding file for targeted skin color filtering.",
)
@click.option(
    "--recursive",
    "-R",
    is_flag=True,
    default=False,
    help="Search subdirectories recursively.",
)
@click.option(
    "--limit",
    "-n",
    type=int,
    default=None,
    help="Maximum number of passing images to copy.",
)
@click.option(
    "--log",
    "-l",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write detailed processing log to file.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show prefilter scores for each image.",
)
def prefilter(
    search_dir: Path | None,
    output_dir: Path,
    encoding: Path | None,
    recursive: bool,
    limit: int | None,
    log: Path | None,
    verbose: bool,
) -> None:
    """Copy images that pass prefilter (likely contain people) to output directory.

    This command runs only the fast prefilter (no face detection) to quickly
    filter a large photo collection to images likely containing people.

    Without --encoding: uses generic skin-tone detection.
    With --encoding: uses targeted filtering based on reference person's skin color.

    Input modes:
    - With --search-dir: scans the directory for images
    - Without --search-dir: reads image paths from stdin (one per line)

    Output modes:
    - Non-verbose (default): prints every processed image path to stdout (enables resumability)
    - Verbose (-v): prints detailed status for each image
    - With --log: writes detailed log to file regardless of verbose mode

    Passing images are copied to --output-dir regardless of output mode.
    """
    from typing import Iterator

    # Helper to write to log file
    log_file = None
    if log:
        log_file = open(log, "w")

    def log_write(msg: str) -> None:
        """Write message to log file if open."""
        if log_file:
            log_file.write(msg + "\n")
            log_file.flush()

    def info_output(msg: str) -> None:
        """Write informational message to appropriate destination."""
        if verbose:
            click.echo(msg)
        log_write(msg)

    # Load skin info if encoding provided
    skin_info: dict | None = None
    if encoding is not None:
        info_output(f"Loading encoding from: {encoding}")
        _, skin_info = load_encoding(encoding)
        if skin_info:
            r, g, b = skin_info["median_rgb"]
            info_output(f"Targeted skin filter: enabled (RGB: {r}, {g}, {b})")
        else:
            info_output("Targeted skin filter: disabled (encoding has no skin data)")
    else:
        info_output("Using generic prefilter (no encoding provided)")

    # Determine input source
    using_stdin = search_dir is None
    if using_stdin:
        info_output("Reading image paths from stdin...")
        image_source: Path | Iterator[Path] = iter_paths_from_stdin()
    else:
        info_output(f"Scanning images in: {search_dir}")
        if recursive:
            info_output("(recursive mode enabled)")
        image_source = search_dir

    info_output("")

    # Track progress
    scanned = 0
    passed_count = 0
    errors_found = 0

    def format_prefilter(pf) -> str:
        """Format prefilter scores for display."""
        parts = []
        if pf.entropy is not None:
            parts.append(f"entropy={pf.entropy:.1f}")
        if pf.reference_skin_percentage is not None:
            parts.append(f"ref_skin={pf.reference_skin_percentage:.1f}%")
        elif pf.skin_percentage is not None:
            parts.append(f"skin={pf.skin_percentage:.1f}%")
        if pf.blob_aspect_ratio is not None:
            parts.append(f"aspect={pf.blob_aspect_ratio:.2f}")
        return ", ".join(parts)

    def update_progress(
        image_path: Path,
        passed: bool,
        error: bool = False,
        error_msg: str | None = None,
        prefilter=None,
    ) -> None:
        nonlocal scanned, passed_count, errors_found
        scanned += 1

        if error:
            errors_found += 1
            status = f"ERROR: {error_msg}" if error_msg else "ERROR"
        elif passed:
            passed_count += 1
            if prefilter:
                scores = format_prefilter(prefilter)
                status = f"PASS ({scores})" if scores else "PASS"
            else:
                status = "PASS"
        else:
            if prefilter:
                scores = format_prefilter(prefilter)
                status = f"SKIP: {prefilter.reason} ({scores})" if scores else f"SKIP: {prefilter.reason}"
            else:
                status = "SKIP"

        # Detailed output to verbose stdout or log file
        detail_msg = f"  [{scanned}] {image_path.name} - {status}"
        if verbose:
            click.echo(detail_msg)
        log_write(detail_msg)

        # In non-verbose mode, output every processed image path to stdout
        # (enables resumability - can diff output against input to find unprocessed images)
        if not verbose:
            click.echo(str(image_path))

        sys.stdout.flush()

    # Process images
    info_output("Starting prefilter scan...")
    interrupted = False
    try:
        result = process_images_prefilter_only(
            image_source=image_source,
            output_dir=output_dir,
            recursive=recursive,
            limit=limit,
            progress_callback=update_progress,
            skin_info=skin_info,
        )
        # Sync counts from result
        scanned = result.total_scanned
        passed_count = result.matches_found
        errors_found = result.errors
    except KeyboardInterrupt:
        interrupted = True

    # Build summary
    summary_lines = [
        "",
        "=" * 50,
        "Summary",
        "=" * 50,
        f"  Images scanned: {scanned}",
        f"  Images passed:  {passed_count}",
        f"  Images skipped: {scanned - passed_count - errors_found}",
        f"  Errors:         {errors_found}",
        f"  Output:         {output_dir}",
    ]

    if scanned > 0:
        pass_rate = (passed_count / scanned) * 100
        summary_lines.append(f"  Pass rate:      {pass_rate:.1f}%")

    if scanned == 0:
        summary_lines.append("")
        source_desc = "stdin" if using_stdin else "the search directory"
        summary_lines.append(f"  No images found in {source_desc}.")

    if interrupted:
        summary_lines.append("  (interrupted by user)")
    elif limit is not None and passed_count >= limit:
        summary_lines.append(f"  (stopped early after reaching limit of {limit})")

    # Output summary to appropriate destination
    # In non-verbose mode: summary goes to stderr (to keep stdout clean for piping)
    # In verbose mode: summary goes to stdout
    # Always write to log file if provided
    for line in summary_lines:
        if verbose:
            click.echo(line)
        else:
            click.echo(line, err=True)
        log_write(line)

    # Close log file
    if log_file:
        log_file.close()
        if verbose:
            click.echo(f"  Log file:       {log}")
        else:
            click.echo(f"  Log file:       {log}", err=True)

    sys.stdout.flush()


if __name__ == "__main__":
    main()
