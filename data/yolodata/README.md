# Trash image annotations

Prepared 14,620 images with one main-object bounding box each.

| Split | Images |
| --- | ---: |
| Training Dataset | 9,318 |
| Testing Dataset | 2,651 |
| Validation Dataset | 2,651 |

## Folders

- `annotated/`: photo copies with one red outline, no caption or added padding.
- `images/`: clean images for YOLO training, with no drawn outline.
- `labels/`: matching YOLO text labels, one row per image.
- `data.yaml`: training configuration preserving the existing train/validation/test membership.
- `review.html`: local gallery for inspecting automatic boxes.
- `needs_review.csv`: flagged boxes and inherited class-label issues.
- `annotation_manifest.csv` and `annotations.jsonl`: all source paths, box coordinates, and method details.

## Quality and review

These are automatic annotations. 6,216 images are flagged for review; the others were not individually human-reviewed.
Review the boxes before treating training or evaluation labels as ground truth. A full-frame provisional box means the main object could not be isolated reliably.
Visually inspected 45 representative previews and 190 difficult images, corrected the difficult cases, and applied the same corrections to byte-identical copies. These checks do not establish accuracy for every other image.
The box expands around the chosen object into a square where the image allows it. At image edges it is clipped to the original photo, producing a rectangle when a complete square cannot fit. This preserves the photo without adding margins or cutting out object pixels.
Shoes photographed as a pair may share one main-subject box. Other multi-object scenes are flagged where detected.
Class IDs follow the original folders: 0 Compost, 1 Garbage, 2 None, 3 Recycling. None is an object class here because it contains batteries, clothing, and shoes; it is not treated as an empty scene.
The original labels are preserved, including any pre-existing category conflicts. These are flagged where detected; the detector does not silently relabel the dataset.

## Source preservation

Raw Data remains unchanged. Clean JPEG copies retain the original bytes except for 18 files requiring EXIF-orientation or image-format normalization. These are exported as upright RGB JPEGs. Red outlines are saved only in the separate annotated folder.

## Method

Local YOLOE-26s detects possible object regions. A U2NetP foreground model supplies a fallback for weak or ambiguous detections. The selected box is expanded slightly and converted into a visible square outline. No generative image editing was used for this batch.
[YOLOE documentation](https://docs.ultralytics.com/models/yoloe/) · [U2NetP model source](https://github.com/danielgatis/rembg/blob/main/rembg/sessions/u2netp.py)

Output folder: C:\Users\anshi\Desktop\garbage sorting\data\yolodata

## Final verification

Independent validation passed with zero file or label errors. All 14,620 original source files match their pre-run SHA-256 fingerprints. All 14,620 labels are valid and paired with correctly sized clean images and previews. Checked red borders in 138 sampled previews. The 18 normalized clean copies passed image-content checks; the other 14,602 clean copies are byte-identical to their sources.

These checks verify file integrity and annotation structure; they do not establish that every automatically chosen box is semantically correct.

