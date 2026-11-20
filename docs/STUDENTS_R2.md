# Students round-2 dataset and the run trained on it

This records one training run that was made **outside** the DVC pipeline, so
that it can still be traced to exact inputs.

- Run: `bluespotter_lc_students_r2_20261119_212344` (19–20 Nov 2026)
- MLflow: experiment `bluespotter-lc-segmentation`, run id
  `fa7087e082c64194bc45ef98355a9e6d`, tagged `logged_retrospectively`
- Numbers: `reports/students_r2_run.json`

## What the dataset is

The three students outlined LC neurons in the scoring app (round 2). Their work
is stored as polygons in `scoring/<student>/annotations_r2.jsonl` on Drive, not
as Cellpose mask files, so it is not covered by `train.csv` / `test.csv` and the
`manifest.py` loader cannot read it yet.

Two small files describe it. As everywhere in this repo they hold links and
checksums, never images:

| File (DVC-tracked) | Content |
|---|---|
| `data/students_r2/students_r2_manifest.csv` | one row per image: split, item id, student whose outlines were used, number of outlines, image path below `TeamPrigge/`, Drive id, md5, size and modification time of the image, and the annotation file it came from |
| `data/students_r2/students_r2_annotation_files.csv` | the three `annotations_r2.jsonl` files with Drive id, md5, size and modification time |

`dvc pull data/students_r2` fetches both from the `drivestore` remote.

181 images, 12,987 cells, 5.87 GB of images on Drive. Where several students
outlined the same image, the student with the most outlines was used. The split
is a random 15 % of images (seed 0): 154 train, 27 held out. It is **not**
grouped by animal.

## How the run differs from `bluespotter.train`

- Started from `bluespotter_lc_20260730_140712`, not from `cpsam`.
- Polygons were rasterised to label masks; each image was cropped to the
  outlined cells plus a margin and downscaled so the median cell diameter is
  about 45 px.
- Trained with `cellpose.train.train_seg` in a Colab cell (100 epochs, learning
  rate 1e-5, weight decay 0.1, batch size 1, T4). The code is saved on Drive in
  `BrainsToSlice/NM_analysis/bluespotter_training/code_*.py`.
- No git commit or index hash was attached at training time; the manifest above
  was written the next morning from the same image list.

## Result

Held-out student images, IoU 0.5, 1750 outlined cells:

| model | AP50 | AP75 | found | extra | missed |
|---|---|---|---|---|---|
| `bluespotter_lc_20260730_140712` | 0.605 | 0.276 | 1144 | 365 | 606 |
| this run | 0.653 | 0.377 | 1172 | 248 | 578 |

On the 45 images outlined by more than one student, one student's outlines
match only 52 % of another's under the same rule. Label disagreement, not
training length, limits these scores.

## To bring this into the pipeline properly

1. Teach `manifest.py` a mask type for polygon annotations (read the jsonl,
   rasterise), so `drive_index`, `train` and `evaluate` can use these rows.
2. Group the split by animal.
3. Re-run through `dvc repro` so the run carries commit and dataset hashes.
