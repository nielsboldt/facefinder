"""Command-line interface for Face Finder."""

import sys
from pathlib import Path

import click

from .detector import (
    analyze_reference_images,
    extract_face_thumbnail,
    load_encoding,
    load_reference_encodings,
    save_encoding,
)
from .matcher import process_images, process_images_prefilter_only
from .prefilter import PrefilterResult


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
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory to search for images containing the person.",
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
    "--debug-log",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write debug log to file (useful for diagnosing crashes).",
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
    search_dir: Path,
    output_dir: Path,
    tolerance: float,
    model: str,
    recursive: bool,
    limit: int | None,
    debug_log: Path | None,
    verbose: bool,
) -> None:
    """Find images containing a specific person."""
    # Validate that exactly one of reference_dir or encoding is provided
    if reference_dir is None and encoding is None:
        click.echo("Error: Must provide either --reference-dir or --encoding.", err=True)
        raise SystemExit(1)
    if reference_dir is not None and encoding is not None:
        click.echo("Error: Cannot provide both --reference-dir and --encoding.", err=True)
        raise SystemExit(1)

    # Load encoding from file or analyze reference images
    skin_info: dict | None = None
    if encoding is not None:
        click.echo(f"Loading encoding from: {encoding}")
        sys.stdout.flush()
        reference_encoding, skin_info = load_encoding(encoding)
        click.echo("Encoding loaded successfully.")
        sys.stdout.flush()
    else:
        click.echo(f"Loading reference images from: {reference_dir}")
        click.echo(f"Model: {model}" + (" (better for profiles/side views)" if model == "cnn" else " (fast, frontal faces)"))
        sys.stdout.flush()
        result = load_reference_encodings(reference_dir, tolerance, model=model)

        if result is None:
            click.echo("Error: No faces found in reference images.", err=True)
            raise SystemExit(1)

        reference_encoding, images_with_face, total_images = result
        click.echo(f"Found common face appearing in {images_with_face}/{total_images} reference images.")
        if images_with_face < total_images:
            click.echo(
                f"  Note: Face not found in {total_images - images_with_face} image(s). "
                "These may have no detectable faces or contain different people."
            )
        sys.stdout.flush()

    # Display targeted prefilter status
    if skin_info and model in ("cnn", "auto"):
        r, g, b = skin_info["median_rgb"]
        click.echo(f"Targeted skin prefilter: enabled (RGB: {r}, {g}, {b})")
    elif model in ("cnn", "auto"):
        click.echo("Targeted skin prefilter: disabled (using generic heuristic)")
    sys.stdout.flush()

    # Scan images (streaming - no pre-count for large directories)
    click.echo(f"Scanning images in: {search_dir}")
    if recursive:
        click.echo("(recursive mode enabled)")
    click.echo("")
    sys.stdout.flush()

    # Track progress with per-image logging
    scanned = 0
    matches_found = 0
    errors_found = 0

    # Open debug log file if requested
    debug_file = None
    if debug_log:
        debug_file = open(debug_log, "w")
        debug_file.write(f"Face Finder Debug Log\n")
        debug_file.write(f"Search dir: {search_dir}\n")
        debug_file.write(f"Model: {model}\n")
        debug_file.write(f"{'=' * 50}\n")
        debug_file.flush()

    def log_before_process(image_path: Path, model_used: str) -> None:
        """Log BEFORE processing each image (critical for crash diagnosis)."""
        msg = f"[PROCESSING] {image_path} model={model_used}"
        click.echo(msg)
        sys.stdout.flush()
        if debug_file:
            debug_file.write(msg + "\n")
            debug_file.flush()

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
        click.echo(msg)
        sys.stdout.flush()
        if debug_file:
            debug_file.write(msg + "\n")
            debug_file.flush()

    def update_progress(image_path: Path, matched: bool, error: bool = False, error_msg: str = None, match_info=None) -> None:
        nonlocal scanned, matches_found, errors_found
        scanned += 1
        if error:
            errors_found += 1
            status = f"ERROR: {error_msg}" if error_msg else "ERROR"
        elif matched:
            matches_found += 1
            if verbose and match_info:
                scores = format_prefilter_scores(match_info)
                if scores:
                    status = f"MATCH (distance={match_info.min_distance:.2f}, faces={match_info.num_faces}, {scores})"
                else:
                    status = f"MATCH (distance={match_info.min_distance:.2f}, faces={match_info.num_faces})"
            else:
                status = "MATCH"
        else:
            if verbose and match_info:
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
        msg = f"  [{scanned}] {image_path.name} - {status}"
        click.echo(msg)
        sys.stdout.flush()
        if debug_file:
            debug_file.write(msg + "\n")
            debug_file.flush()

    # Process images
    click.echo("Starting image processing...")
    sys.stdout.flush()
    interrupted = False
    try:
        result = process_images(
            search_dir=search_dir,
            output_dir=output_dir,
            reference_encoding=reference_encoding,
            tolerance=tolerance,
            recursive=recursive,
            limit=limit,
            progress_callback=update_progress,
            pre_process_callback=log_before_process,
            model=model,
            verbose=verbose,
            prefilter_callback=log_prefilter if verbose else None,
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

    # Print summary
    click.echo("")
    click.echo("=" * 50)
    click.echo("Summary")
    click.echo("=" * 50)
    click.echo(f"  Images scanned: {result.total_scanned}")
    click.echo(f"  Matches found:  {result.matches_found}")
    click.echo(f"  Errors:         {result.errors}")
    click.echo(f"  Output:         {output_dir}")

    if result.total_scanned == 0:
        click.echo("")
        click.echo("  No images found in the search directory.")

    if interrupted:
        click.echo("  (interrupted by user)")
    elif limit is not None and result.matches_found >= limit:
        click.echo(f"  (stopped early after reaching limit of {limit})")

    # Close debug log file
    if debug_file:
        debug_file.write(f"{'=' * 50}\n")
        debug_file.write(f"Completed: scanned={result.total_scanned} matches={result.matches_found} errors={result.errors}\n")
        debug_file.close()
        click.echo(f"  Debug log: {debug_log}")

    sys.stdout.flush()


@main.command()
@click.option(
    "--search-dir",
    "-s",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory to search for images.",
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
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show prefilter scores for each image.",
)
def prefilter(
    search_dir: Path,
    output_dir: Path,
    encoding: Path | None,
    recursive: bool,
    limit: int | None,
    verbose: bool,
) -> None:
    """Copy images that pass prefilter (likely contain people) to output directory.

    This command runs only the fast prefilter (no face detection) to quickly
    filter a large photo collection to images likely containing people.

    Without --encoding: uses generic skin-tone detection.
    With --encoding: uses targeted filtering based on reference person's skin color.
    """
    # Load skin info if encoding provided
    skin_info: dict | None = None
    if encoding is not None:
        click.echo(f"Loading encoding from: {encoding}")
        sys.stdout.flush()
        _, skin_info = load_encoding(encoding)
        if skin_info:
            r, g, b = skin_info["median_rgb"]
            click.echo(f"Targeted skin filter: enabled (RGB: {r}, {g}, {b})")
        else:
            click.echo("Targeted skin filter: disabled (encoding has no skin data)")
    else:
        click.echo("Using generic prefilter (no encoding provided)")
    sys.stdout.flush()

    # Scan info
    click.echo(f"Scanning images in: {search_dir}")
    if recursive:
        click.echo("(recursive mode enabled)")
    click.echo("")
    sys.stdout.flush()

    # Track progress
    scanned = 0
    passed_count = 0
    errors_found = 0

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
            if verbose and prefilter:
                scores = format_prefilter(prefilter)
                status = f"PASS ({scores})" if scores else "PASS"
            else:
                status = "PASS"
        else:
            if verbose and prefilter:
                scores = format_prefilter(prefilter)
                status = f"SKIP: {prefilter.reason} ({scores})" if scores else f"SKIP: {prefilter.reason}"
            else:
                status = "SKIP"

        click.echo(f"  [{scanned}] {image_path.name} - {status}")
        sys.stdout.flush()

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

    # Process images
    click.echo("Starting prefilter scan...")
    sys.stdout.flush()
    interrupted = False
    try:
        result = process_images_prefilter_only(
            search_dir=search_dir,
            output_dir=output_dir,
            recursive=recursive,
            limit=limit,
            progress_callback=update_progress if verbose else None,
            skin_info=skin_info,
        )
        # If not verbose, we still need the counts
        if not verbose:
            scanned = result.total_scanned
            passed_count = result.matches_found
            errors_found = result.errors
    except KeyboardInterrupt:
        interrupted = True
        # Use tracked progress for partial results
        result = None

    # Print summary
    click.echo("")
    click.echo("=" * 50)
    click.echo("Summary")
    click.echo("=" * 50)
    click.echo(f"  Images scanned: {scanned}")
    click.echo(f"  Images passed:  {passed_count}")
    click.echo(f"  Images skipped: {scanned - passed_count - errors_found}")
    click.echo(f"  Errors:         {errors_found}")
    click.echo(f"  Output:         {output_dir}")

    if scanned > 0:
        pass_rate = (passed_count / scanned) * 100
        click.echo(f"  Pass rate:      {pass_rate:.1f}%")

    if scanned == 0:
        click.echo("")
        click.echo("  No images found in the search directory.")

    if interrupted:
        click.echo("  (interrupted by user)")
    elif limit is not None and passed_count >= limit:
        click.echo(f"  (stopped early after reaching limit of {limit})")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
