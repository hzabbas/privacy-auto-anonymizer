"""
Secure Image Anonymization and Redaction Filters Module.
Privacy-Auto-Anonymizer Project.

This module provides privacy-preserving redaction algorithms designed to resist
modern adversarial inversion attacks (e.g., Diffusion model unblurring, deep
super-resolution, and contextual reconstruction).

Scientific Grounding & Theoretical Citations (see RESEARCH_PAPERS.md):
1. Explainability-Driven Incremental Image Anonymization (2025):
   Differential Privacy-inspired Pixelation (DP-Pix):
   I_anonymized = clip(I_pixelated + N(0, sigma^2), 0, 255)
2. Privacy Blur (2025, arXiv:2512.16086) & Revelio (2025, arXiv:2506.12344):
   Demonstrates that conventional Gaussian Blur is up to 95.9% invertible by
   conditional diffusion models. Gaussian blur is retained strictly for legacy/visual
   purposes with explicit security warnings.
3. RedactionBench (2026, arXiv:2606.18782):
   Contextual Integrity framework for sensitive textual and numeric entities,
   mandating zero-entropy Solid Black Box masking to eliminate visual signal leakage.
"""

from typing import Union, List, Tuple
import numpy as np
import cv2


def dp_pix_filter(
    image_crop: np.ndarray,
    pixel_size: int = 16,
    noise_scale: float = 8.0
) -> np.ndarray:
    """
    Applies Differential Privacy-inspired Pixelation (DP-Pix) with additive Gaussian noise.

    Mathematical Formulation (Incremental Image Anonymization, 2025):
        I_pixelated = Upsample(Downsample(I, ratio=1/pixel_size), method=INTER_NEAREST)
        I_anonymized = clip(I_pixelated + N(0, noise_scale^2), 0, 255)

    The combined mosaic downsampling breaks high-frequency geometric priors, while the
    additive stochastic perturbation destroys deterministic gradient and diffusion inversion
    pathways.

    Args:
        image_crop: BGR or Grayscale image patch (H, W, C) or (H, W) as np.uint8.
        pixel_size: Cell size for mosaic binning (default: 16 pixels).
        noise_scale: Standard deviation (sigma) of additive Gaussian noise (default: 8.0).

    Returns:
        Perturbed, anonymized image crop of identical shape and np.uint8 dtype.
    """
    if image_crop.size == 0:
        return image_crop

    h, w = image_crop.shape[:2]
    if h <= 0 or w <= 0:
        return image_crop

    # Ensure valid reduction dimensions
    pixel_size = max(2, int(pixel_size))
    down_w = max(1, w // pixel_size)
    down_h = max(1, h // pixel_size)

    # 1. Spatial aggregation (downsample -> upsample with nearest neighbor)
    small = cv2.resize(image_crop, (down_w, down_h), interpolation=cv2.INTER_NEAREST)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    # 2. Stochastic noise injection to mitigate deep generative de-anonymization
    if noise_scale > 0:
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=pixelated.shape)
        anonymized = np.clip(pixelated.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    else:
        anonymized = pixelated

    return anonymized


def solid_black_box(image_crop: np.ndarray) -> np.ndarray:
    """
    Applies zero-entropy Solid Black Box masking for confidential textual and tabular data.

    Theoretical Justification (RedactionBench, 2026 - arXiv:2606.18782):
        Under Contextual Integrity for textual documents and identity tokens, any
        continuous or blurred representation leaves character boundary and kerning
        artifacts exploitable by multimodal OCR. Complete zero-filling (all channels = 0)
        guarantees mathematical information entropy H(X) = 0 over the redacted region.

    Args:
        image_crop: Target image region (H, W, C) or (H, W).

    Returns:
        All-black array of identical shape and np.uint8 dtype.
    """
    if image_crop.size == 0:
        return image_crop

    return np.zeros_like(image_crop)


def heavy_gaussian_blur(image_crop: np.ndarray, ksize: int = 51) -> np.ndarray:
    """
    Applies heavy Gaussian spatial filtering to an image crop.

    SECURITY WARNING (Privacy Blur arXiv:2512.16086 / Revelio arXiv:2506.12344):
        Gaussian blur is an invertible low-pass linear operation. State-of-the-art
        diffusion-based reverse models can reconstruct original facial and license features
        from Gaussian-blurred images with up to 95.9% fidelity.
        Use `dp_pix_filter` or `solid_black_box` when strict privacy preservation is required.

    Args:
        image_crop: Target image region (H, W, C) or (H, W).
        ksize: Gaussian kernel size (must be an odd integer >= 3, default: 51).

    Returns:
        Spatially blurred image crop.
    """
    if image_crop.size == 0:
        return image_crop

    # Kernel size must be positive and odd
    k = max(3, int(ksize))
    if k % 2 == 0:
        k += 1

    # Kernel cannot exceed region dimensions
    h, w = image_crop.shape[:2]
    max_k = min(h, w)
    if max_k < k:
        k = max_k if max_k % 2 != 0 else max(1, max_k - 1)

    if k <= 1:
        return image_crop

    return cv2.GaussianBlur(image_crop, (k, k), sigmaX=0)


def apply_redaction(
    image: np.ndarray,
    bbox: Union[List[int], Tuple[int, int, int, int]],
    filter_type: str = "dp_pix",
    copy: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Applies privacy redaction to a specified bounding box within a host image.

    Coordinates are safely clamped to image boundaries [0, W] and [0, H] to prevent
    out-of-bounds index errors.

    Args:
        image: Host image (OpenCV BGR ndarray of shape H, W, C).
        bbox: Bounding box coordinates in [x1, y1, x2, y2] format.
        filter_type: Redaction technique:
                     - "dp_pix": Differential Privacy pixelation + noise (faces, plates)
                     - "solid_black": Absolute zero-out mask (text, sensitive tokens)
                     - "blur": Conventional heavy Gaussian blur (visual only)
        copy: If True, returns a new image copy without modifying input in-place.
        **kwargs: Optional hyperparameter overrides passed to the selected filter:
                  e.g., pixel_size, noise_scale, ksize.

    Returns:
        Redacted image as a NumPy array.
    """
    if image is None or image.size == 0:
        return image

    target_img = image.copy() if copy else image
    height, width = target_img.shape[:2]

    # Unpack and clamp coordinates
    raw_x1, raw_y1, raw_x2, raw_y2 = bbox
    x1 = max(0, min(int(raw_x1), width))
    y1 = max(0, min(int(raw_y1), height))
    x2 = max(0, min(int(raw_x2), width))
    y2 = max(0, min(int(raw_y2), height))

    # Validate non-empty bounding box
    if x2 <= x1 or y2 <= y1:
        return target_img

    crop = target_img[y1:y2, x1:x2]
    if crop.size == 0:
        return target_img

    # Select and apply filter
    filter_lower = filter_type.strip().lower()
    if filter_lower in ("dp_pix", "pixelate", "mosaic"):
        pixel_size = kwargs.get("pixel_size", 16)
        noise_scale = kwargs.get("noise_scale", 8.0)
        redacted_crop = dp_pix_filter(crop, pixel_size=pixel_size, noise_scale=noise_scale)
    elif filter_lower in ("solid_black", "solid", "black", "black_box"):
        redacted_crop = solid_black_box(crop)
    elif filter_lower in ("blur", "gaussian", "gaussian_blur"):
        ksize = kwargs.get("ksize", 51)
        redacted_crop = heavy_gaussian_blur(crop, ksize=ksize)
    else:
        raise ValueError(
            f"Unsupported filter_type '{filter_type}'. "
            f"Available options: 'dp_pix', 'solid_black', 'blur'"
        )

    # Re-insert redacted crop into image
    target_img[y1:y2, x1:x2] = redacted_crop
    return target_img


if __name__ == "__main__":
    print("--- Running Anonymization Filters Self-Test ---")

    # 1. Create a synthetic test image with multiple sensitive zones
    canvas = np.full((400, 600, 3), 200, dtype=np.uint8)

    # Add simulated face area (gradients and colors)
    cv2.circle(canvas, (150, 150), 70, (180, 150, 220), -1)
    cv2.circle(canvas, (130, 130), 10, (50, 50, 50), -1)
    cv2.circle(canvas, (170, 130), 10, (50, 50, 50), -1)

    # Add simulated license plate
    cv2.rectangle(canvas, (350, 80), (550, 150), (255, 255, 255), -1)
    cv2.putText(canvas, "IR-24 991B77", (360, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Add confidential text
    cv2.putText(canvas, "TOP SECRET CLASSIFIED", (100, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 120), 3)

    # 2. Test solid_black_box on text
    text_bbox = [90, 280, 560, 340]
    apply_redaction(canvas, text_bbox, filter_type="solid_black")
    text_area = canvas[280:340, 90:560]
    assert np.all(text_area == 0), "Solid black box test failed (contains non-zero pixels)"
    print("[PASS] solid_black_box completely zeroed text area.")

    # 3. Test dp_pix_filter on simulated face
    face_bbox = [80, 80, 220, 220]
    orig_face = canvas[80:220, 80:220].copy()
    apply_redaction(canvas, face_bbox, filter_type="dp_pix", pixel_size=16, noise_scale=10.0)
    redacted_face = canvas[80:220, 80:220]
    assert not np.array_equal(orig_face, redacted_face), "DP-Pix face filter produced identical image"
    print("[PASS] dp_pix_filter successfully applied differential privacy pixelation + noise.")

    # 4. Test heavy_gaussian_blur on simulated plate
    plate_bbox = [340, 70, 560, 160]
    orig_plate = canvas[70:160, 340:560].copy()
    apply_redaction(canvas, plate_bbox, filter_type="blur", ksize=41)
    blurred_plate = canvas[70:160, 340:560]
    assert not np.array_equal(orig_plate, blurred_plate), "Gaussian blur produced identical image"
    print("[PASS] heavy_gaussian_blur successfully applied.")

    # 5. Test out-of-bounds boundary clamping
    oob_bbox = [-50, -30, 700, 500]
    apply_redaction(canvas, oob_bbox, filter_type="solid_black")
    assert np.all(canvas == 0), "Clamping test failed on out-of-bounds bounding box"
    print("[PASS] Out-of-bounds coordinates clamped successfully without exception.")

    print("\n--- All Filter Tests Passed Successfully! ---")
