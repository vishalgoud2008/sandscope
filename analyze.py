import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt


# =========================================================
# SandScope v2.0
# Plain Black Square Calibration
# =========================================================

INPUT_FOLDER = "images"
RESULTS_FOLDER = "results"

# Physical size of the black calibration square.
# Measure the real square with a ruler.
REFERENCE_SIZE_MM = 25.0

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)

os.makedirs(
    RESULTS_FOLDER,
    exist_ok=True
)


# =========================================================
# FIND BLACK CALIBRATION SQUARE
# =========================================================

def find_black_square(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    height, width = gray.shape

    image_area = height * width

    # -----------------------------------------------------
    # Find very dark regions
    # -----------------------------------------------------

    _, dark = cv2.threshold(
        gray,
        60,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Clean noise
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        dark,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        # Reject tiny objects
        if area < image_area * 0.002:
            continue

        # Reject huge dark regions
        if area > image_area * 0.20:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter <= 0:
            continue

        # Approximate shape
        approx = cv2.approxPolyDP(
            contour,
            0.03 * perimeter,
            True
        )

        # We want a quadrilateral
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(
            approx
        )

        if w <= 0 or h <= 0:
            continue

        aspect_ratio = w / h

        # Approximately square
        if not 0.85 <= aspect_ratio <= 1.15:
            continue

        # -------------------------------------------------
        # Check rotated rectangle
        # -------------------------------------------------

        rect = cv2.minAreaRect(
            contour
        )

        (_, _), (rw, rh), _ = rect

        if rw <= 0 or rh <= 0:
            continue

        rotated_ratio = (
            max(rw, rh) /
            min(rw, rh)
        )

        if rotated_ratio > 1.15:
            continue

        # -------------------------------------------------
        # Check darkness inside square
        # -------------------------------------------------

        mask = np.zeros_like(
            gray
        )

        cv2.drawContours(
            mask,
            [approx],
            -1,
            255,
            -1
        )

        mean_inside = cv2.mean(
            gray,
            mask=mask
        )[0]

        if mean_inside > 70:
            continue

        # -------------------------------------------------
        # Score candidate
        # -------------------------------------------------

        square_quality = (
            area /
            (
                max(w, h) *
                max(w, h)
            )
        )

        candidates.append(
            (
                square_quality,
                area,
                approx
            )
        )

    if not candidates:
        return None

    # Best square candidate
    candidates.sort(
        key=lambda item: (
            item[0],
            item[1]
        ),
        reverse=True
    )

    return candidates[0][2]


# =========================================================
# CALCULATE SCALE
# =========================================================

def calculate_scale(square):

    points = square.reshape(
        4,
        2
    ).astype(
        np.float32
    )

    side_lengths = []

    for i in range(4):

        p1 = points[i]

        p2 = points[
            (i + 1) % 4
        ]

        distance = np.linalg.norm(
            p2 - p1
        )

        side_lengths.append(
            distance
        )

    side_lengths = np.array(
        side_lengths
    )

    pixel_size = float(
        np.mean(side_lengths)
    )

    if pixel_size <= 0:
        return None

    pixels_per_mm = (
        pixel_size /
        REFERENCE_SIZE_MM
    )

    mm_per_pixel = (
        REFERENCE_SIZE_MM /
        pixel_size
    )

    return (
        pixel_size,
        pixels_per_mm,
        mm_per_pixel
    )


# =========================================================
# DETECT GRAINS
# =========================================================

def detect_grains(
    image,
    square
):

    height, width = image.shape[:2]

    # -----------------------------------------------------
    # Mask black calibration square
    # -----------------------------------------------------

    square_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    cv2.fillPoly(
        square_mask,
        [
            square.astype(
                np.int32
            )
        ],
        255
    )

    # -----------------------------------------------------
    # LAB processing
    # -----------------------------------------------------

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    lightness = lab[:, :, 0]

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        lightness
    )

    blur = cv2.GaussianBlur(
        enhanced,
        (5, 5),
        0
    )

    # -----------------------------------------------------
    # Threshold
    # -----------------------------------------------------

    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    # Remove calibration square
    binary[
        square_mask == 255
    ] = 0

    # -----------------------------------------------------
    # Morphological cleaning
    # -----------------------------------------------------

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    opening = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    opening = cv2.morphologyEx(
        opening,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    # -----------------------------------------------------
    # Background
    # -----------------------------------------------------

    sure_background = cv2.dilate(
        opening,
        kernel,
        iterations=3
    )

    # -----------------------------------------------------
    # Distance transform
    # -----------------------------------------------------

    distance = cv2.distanceTransform(
        opening,
        cv2.DIST_L2,
        5
    )

    if distance.max() == 0:

        return [], 0

    # -----------------------------------------------------
    # Grain centers
    # -----------------------------------------------------

    _, sure_foreground = cv2.threshold(
        distance,
        0.30 * distance.max(),
        255,
        0
    )

    sure_foreground = np.uint8(
        sure_foreground
    )

    # -----------------------------------------------------
    # Unknown region
    # -----------------------------------------------------

    unknown = cv2.subtract(
        sure_background,
        sure_foreground
    )

    # -----------------------------------------------------
    # Watershed markers
    # -----------------------------------------------------

    number, markers = cv2.connectedComponents(
        sure_foreground
    )

    initial_markers = number - 1

    markers = markers + 1

    markers[
        unknown == 255
    ] = 0

    # -----------------------------------------------------
    # Watershed
    # -----------------------------------------------------

    markers = cv2.watershed(
        image,
        markers
    )

    grains = []

    # -----------------------------------------------------
    # Extract grain regions
    # -----------------------------------------------------

    for marker_id in range(
        2,
        markers.max() + 1
    ):

        mask = (
            np.uint8(
                markers == marker_id
            ) * 255
        )

        pixel_area = cv2.countNonZero(
            mask
        )

        # Ignore tiny detections
        if pixel_area < 20:
            continue

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        contour = max(
            contours,
            key=cv2.contourArea
        )

        area = cv2.contourArea(
            contour
        )

        if area < 20:
            continue

        # Equivalent circular diameter
        diameter_pixels = float(
            np.sqrt(
                (4.0 * area) /
                np.pi
            )
        )

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter > 0:

            circularity = float(
                (
                    4.0 *
                    np.pi *
                    area
                ) /
                (
                    perimeter ** 2
                )
            )

        else:

            circularity = 0.0

        moments = cv2.moments(
            contour
        )

        if moments["m00"] != 0:

            cx = int(
                moments["m10"] /
                moments["m00"]
            )

            cy = int(
                moments["m01"] /
                moments["m00"]
            )

        else:

            cx = 0
            cy = 0

        grains.append({
            "area": area,
            "diameter_pixels": diameter_pixels,
            "circularity": circularity,
            "cx": cx,
            "cy": cy,
            "contour": contour
        })

    return (
        grains,
        initial_markers
    )


# =========================================================
# PROCESS ONE IMAGE
# =========================================================

def process_image(filename):

    print()
    print("========================================")
    print("Processing:", filename)
    print("========================================")

    image_path = os.path.join(
        INPUT_FOLDER,
        filename
    )

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "ERROR: Could not read image."
        )

        return None

    print(
        "Image size:",
        image.shape
    )

    # -----------------------------------------------------
    # FIND BLACK SQUARE
    # -----------------------------------------------------

    print()
    print(
        "STEP 1 - CALIBRATION"
    )

    square = find_black_square(
        image
    )

    if square is None:

        print(
            "CALIBRATION FAILED"
        )

        print(
            "No valid 25 mm black square found."
        )

        print(
            "Skipping this sample."
        )

        return None

    scale = calculate_scale(
        square
    )

    if scale is None:

        print(
            "ERROR: Could not calculate scale."
        )

        return None

    (
        square_pixels,
        pixels_per_mm,
        mm_per_pixel
    ) = scale

    print(
        "CALIBRATION SUCCESSFUL"
    )

    print(
        f"Reference: "
        f"{REFERENCE_SIZE_MM:.2f} mm"
    )

    print(
        f"Square in image: "
        f"{square_pixels:.2f} pixels"
    )

    print(
        f"Scale: "
        f"{pixels_per_mm:.4f} pixels/mm"
    )

    print(
        f"Scale: "
        f"{mm_per_pixel:.6f} mm/pixel"
    )

    # -----------------------------------------------------
    # GRAIN ANALYSIS
    # -----------------------------------------------------

    print()
    print(
        "STEP 2 - GRAIN ANALYSIS"
    )

    grains, initial_markers = detect_grains(
        image,
        square
    )

    print(
        "Initial grain markers:",
        initial_markers
    )

    print(
        "Detected grain candidates:",
        len(grains)
    )

    if not grains:

        print(
            "No grain candidates found."
        )

        return None

    # -----------------------------------------------------
    # Convert grain sizes to mm
    # -----------------------------------------------------

    diameters_mm = np.array([
        grain["diameter_pixels"] *
        mm_per_pixel
        for grain in grains
    ])

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    mean_mm = float(
        np.mean(diameters_mm)
    )

    median_mm = float(
        np.median(diameters_mm)
    )

    std_mm = float(
        np.std(diameters_mm)
    )

    d10 = float(
        np.percentile(
            diameters_mm,
            10
        )
    )

    d30 = float(
        np.percentile(
            diameters_mm,
            30
        )
    )

    d50 = float(
        np.percentile(
            diameters_mm,
            50
        )
    )

    d60 = float(
        np.percentile(
            diameters_mm,
            60
        )
    )

    uniformity = (
        d60 / d10
        if d10 > 0
        else 0
    )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print()
    print("========================================")
    print("CALIBRATED GRAIN RESULTS")
    print("========================================")

    print(
        f"Mean diameter:      {mean_mm:.4f} mm"
    )

    print(
        f"Median diameter:    {median_mm:.4f} mm"
    )

    print(
        f"Standard deviation: {std_mm:.4f} mm"
    )

    print(
        f"D10:                {d10:.4f} mm"
    )

    print(
        f"D30:                {d30:.4f} mm"
    )

    print(
        f"D50:                {d50:.4f} mm"
    )

    print(
        f"D60:                {d60:.4f} mm"
    )

    print(
        f"Uniformity:         {uniformity:.3f}"
    )

    # -----------------------------------------------------
    # Create result folder
    # -----------------------------------------------------

    sample_name = os.path.splitext(
        filename
    )[0]

    sample_folder = os.path.join(
        RESULTS_FOLDER,
        sample_name
    )

    os.makedirs(
        sample_folder,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Draw result
    # -----------------------------------------------------

    result = image.copy()

    # Grain outlines
    for grain in grains:

        cv2.drawContours(
            result,
            [grain["contour"]],
            -1,
            (0, 255, 0),
            1
        )

        cv2.circle(
            result,
            (
                grain["cx"],
                grain["cy"]
            ),
            2,
            (0, 0, 255),
            -1
        )

    # Calibration square
    cv2.polylines(
        result,
        [
            square.astype(
                np.int32
            )
        ],
        True,
        (255, 0, 255),
        4
    )

    # -----------------------------------------------------
    # Display information
    # -----------------------------------------------------

    cv2.putText(
        result,
        f"Grains: {len(grains)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2
    )

    cv2.putText(
        result,
        f"D50: {d50:.3f} mm",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2
    )

    cv2.putText(
        result,
        "CALIBRATION: VALID",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 0, 255),
        2
    )

    # -----------------------------------------------------
    # Save original
    # -----------------------------------------------------

    cv2.imwrite(
        os.path.join(
            sample_folder,
            "original.jpg"
        ),
        image
    )

    # -----------------------------------------------------
    # Save analyzed image
    # -----------------------------------------------------

    cv2.imwrite(
        os.path.join(
            sample_folder,
            "analyzed.jpg"
        ),
        result
    )

    # -----------------------------------------------------
    # Save measurements
    # -----------------------------------------------------

    measurements_path = os.path.join(
        sample_folder,
        "measurements.csv"
    )

    with open(
        measurements_path,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "Grain ID",
            "Area pixels^2",
            "Diameter pixels",
            "Diameter mm",
            "Circularity",
            "Center X",
            "Center Y"
        ])

        for index, grain in enumerate(
            grains,
            start=1
        ):

            diameter_mm = (
                grain["diameter_pixels"] *
                mm_per_pixel
            )

            writer.writerow([
                index,
                round(
                    grain["area"],
                    2
                ),
                round(
                    grain["diameter_pixels"],
                    2
                ),
                round(
                    diameter_mm,
                    4
                ),
                round(
                    grain["circularity"],
                    3
                ),
                grain["cx"],
                grain["cy"]
            ])

    # -----------------------------------------------------
    # Save summary
    # -----------------------------------------------------

    summary_path = os.path.join(
        sample_folder,
        "summary.csv"
    )

    with open(
        summary_path,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "Parameter",
            "Value",
            "Unit"
        ])

        writer.writerow([
            "Calibration status",
            "VALID",
            ""
        ])

        writer.writerow([
            "Reference type",
            "Plain black square",
            ""
        ])

        writer.writerow([
            "Reference size",
            REFERENCE_SIZE_MM,
            "mm"
        ])

        writer.writerow([
            "Reference pixels",
            round(
                square_pixels,
                2
            ),
            "pixels"
        ])

        writer.writerow([
            "Pixels per mm",
            round(
                pixels_per_mm,
                5
            ),
            "pixels/mm"
        ])

        writer.writerow([
            "Detected grain candidates",
            len(grains),
            "count"
        ])

        writer.writerow([
            "Mean diameter",
            round(
                mean_mm,
                4
            ),
            "mm"
        ])

        writer.writerow([
            "Median diameter",
            round(
                median_mm,
                4
            ),
            "mm"
        ])

        writer.writerow([
            "Standard deviation",
            round(
                std_mm,
                4
            ),
            "mm"
        ])

        writer.writerow([
            "D10",
            round(
                d10,
                4
            ),
            "mm"
        ])

        writer.writerow([
            "D30",
            round(
                d30,
                4
            ),
            "mm"
        ])

        writer.writerow([
            "D50",
            round(
                d50,
                4
            ),
            "mm"
        ])

        writer.writerow([
            "D60",
            round(
                d60,
                4
            ),
            "mm"
        ])

        writer.writerow([
            "Uniformity coefficient",
            round(
                uniformity,
                3
            ),
            ""
        ])

    # -----------------------------------------------------
    # Histogram
    # -----------------------------------------------------

    histogram_path = os.path.join(
        sample_folder,
        "grain_size_distribution.png"
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.hist(
        diameters_mm,
        bins=20,
        edgecolor="black"
    )

    plt.xlabel(
        "Equivalent grain diameter (mm)"
    )

    plt.ylabel(
        "Number of detected grains"
    )

    plt.title(
        "Image-Derived Sand Grain Size Distribution"
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        histogram_path,
        dpi=150
    )

    plt.close()

    return {
        "Sample": filename,
        "Grains": len(grains),
        "Mean mm": round(
            mean_mm,
            4
        ),
        "D10 mm": round(
            d10,
            4
        ),
        "D30 mm": round(
            d30,
            4
        ),
        "D50 mm": round(
            d50,
            4
        ),
        "D60 mm": round(
            d60,
            4
        )
    }


# =========================================================
# MAIN
# =========================================================

def main():

    files = [
        filename
        for filename in os.listdir(
            INPUT_FOLDER
        )
        if filename.lower().endswith(
            VALID_EXTENSIONS
        )
    ]

    if not files:

        print(
            "No images found in images/"
        )

        return

    print()
    print(
        "Found",
        len(files),
        "image(s)."
    )

    results = []

    for filename in sorted(files):

        result = process_image(
            filename
        )

        if result is not None:

            results.append(
                result
            )

    # -----------------------------------------------------
    # Comparison table
    # -----------------------------------------------------

    if results:

        comparison_path = os.path.join(
            RESULTS_FOLDER,
            "all_samples_comparison.csv"
        )

        fieldnames = [
            "Sample",
            "Grains",
            "Mean mm",
            "D10 mm",
            "D30 mm",
            "D50 mm",
            "D60 mm"
        ]

        with open(
            comparison_path,
            "w",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                results
            )

        print()
        print("========================================")
        print("ALL SAMPLES COMPLETE")
        print("========================================")

        print(
            "Successfully analyzed:",
            len(results)
        )

        print(
            "Comparison:",
            comparison_path
        )

    else:

        print()
        print(
            "No calibrated samples were analyzed."
        )


if __name__ == "__main__":

    main()