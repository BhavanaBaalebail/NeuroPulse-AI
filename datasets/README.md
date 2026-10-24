# External vision datasets

Raw downloads belong in `data/raw_external/` (gitignored). Derived crops belong in `data/derived/`. Do not mix them with `data/train/`.

None of the third-party datasets below were downloaded into this repository during the pipeline repair. Access failed, required a login, or required a signed license. Scripts only import files you place locally.

## MRL Eye Dataset

- Page: https://mrl.cs.vsb.cz/eyedataset.html
- Status: publicly described; download link is on that page (zip of infrared eye crops).
- Labels (in filenames): subject, image id, gender, glasses, **eye state** (0 closed / 1 open), reflections, lighting, sensor.
- License / use: follow the dataset page and cite Fusek, “Pupil localization using geodesic distance.”
- Domain gap: **infrared eye crops**, not RGB webcam faces. An MRL-trained open/closed model will not automatically work on this app’s camera without RGB fine-tuning.
- Import after manual download:

```bash
python import_mrl_eyes.py --zip /path/to/mrlEyes_2018_01.zip
# or
python import_mrl_eyes.py --root /path/to/extracted/mrlEyes_2018_01
```

This writes `data/derived/mrl_eyes/{open,closed}/` plus `data/derived/mrl_eyes/provenance.json`.

## YawDD (IEEE DataPort)

- DOI: https://doi.org/10.21227/e1qm-hb90
- Status: **not downloaded**. IEEE DataPort requires an account. Videos are unlabeled; speech vs yawn is identifiable by watching clips.
- Fit: talking / singing / silent / yawning in a car — useful for false-positive analysis of mouth opening.
- Redistribution: research use of the videos; some subjects forbid public screenshots (see participants table).
- After you download AVI files:

```bash
python import_yawdd.py --root /path/to/YawDD
```

The importer records provenance only and does **not** train a yawn network (that needs clip-level labels you create). `YawnClassifierInterface` in `vision/mouth_state.py` is the hook for a future video model.

## NTHU Driver Drowsiness Detection Dataset

- Page: http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/
- Status: **not downloaded**. Access requires emailing a signed Dataset License Agreement to `cvlablai636@my.nthu.edu.tw`.
- Domain: infrared driving videos (day/night, glasses), behaviors include yawning, slow blinks, nodding, talking, laughing.
- After licensed download, keep files under `data/raw_external/nthu_ddd/` and document the license PDF next to them. Do not commit the videos.

## Local webcam stills (this repo)

- `data/train/drowsy/` — 282 images (`img_*` originals and `aug_drowsy_*`).
- `data/train/notdrowsy/` — 284 images (`img_*` and `aug_awake_*`).
- `data/val/*` — empty at audit time.
- Pairing: `img_N` and `aug_drowsy_N` share a source id. `aug_awake_N` does **not** share ids with `img_N` in notdrowsy (overlap was 0), so those augs are grouped only with the same `aug_awake` id.
- Labels are folder names, not verified per-frame eye-closed labels.

```bash
python prepare_vision_splits.py
```
