# Data

## Dataset
Retinal OCT Images (Kermany et al.): CNV, DME, DRUSEN, NORMAL.

- Download used: https://www.kaggle.com/datasets/paultimothymooney/kermany2018
  (listed as CC BY-NC-SA 4.0)
- Original source: https://data.mendeley.com/datasets/rscbjbr9sj/2 (CC BY 4.0)
- Citation: D. Kermany, M. Goldbaum, W. Cai, et al. Identifying medical
  diagnoses and treatable diseases by image-based deep learning.
  Cell, 172(5):1122-1131, 2018. doi:10.1016/j.cell.2018.02.010

Raw images are not stored in git.

## Setup (run from the repo root)

    pip install -r requirements.txt
    python data/download_data.py --config configs/data.yaml

The download needs a Kaggle token in KAGGLE_API_TOKEN.

## splits.csv
Columns: filepath (relative to image_dir), label, patient_id, split.
Use the committed file. To rebuild it:

    python src/data.py --config configs/data.yaml

## Cleaning and split
- Only one of the two dataset copies is used (images are identical).
- 84,484 images, no corrupted files.
- Removed 466 files with conflicting labels and 7,341 extra duplicates.
- 76,677 images remain.
- Patient-level stratified group split, seed 42:
  train 53,673, val 11,502, test 11,502, no shared patient IDs.
- The test set is only for the final evaluation.
