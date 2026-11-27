"""Data loading, cleaning, splitting and preprocessing for the OCT project."""

import hashlib
from pathlib import Path

import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold


def file_hash(path):
    """Return the MD5 fingerprint of a file's bytes."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def index_files(root):
    """Return a table with the relative path, name and hash of every file."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Folder not found: {root!r}")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append({
                "relpath": path.relative_to(root).as_posix(),
                "name": path.name,
                "md5": file_hash(path),
            })
    return pd.DataFrame(rows)


def parse_filename(name):
    """Return (label, patient_id) from CLASS-PATIENTID-IMAGENUMBER.jpeg."""
    parts = Path(name).stem.split("-")
    if len(parts) != 3:
        raise ValueError(f"Unexpected file name: {name}")
    return parts[0], parts[1]


def build_image_table(image_dir):
    """Return one row per image with its path, hash, label and patient ID."""
    table = index_files(image_dir)
    table = table[table["name"].str.endswith(".jpeg")].reset_index(drop=True)
    parsed = [parse_filename(name) for name in table["name"]]
    table["label"] = [label for label, _ in parsed]
    table["patient_id"] = [patient for _, patient in parsed]
    folders = [Path(path).parent.name for path in table["relpath"]]
    if not (table["label"] == folders).all():
        raise ValueError("A file name and its folder disagree on the class")
    return table


def print_class_summary(table):
    """Print images and patients per class, and patients with several classes."""
    summary = table.groupby("label").agg(
        images=("name", "size"), patients=("patient_id", "nunique"))
    print(summary)
    classes = table.groupby("patient_id")["label"].nunique()
    print("Total images:", len(table))
    print("Unique patient IDs:", len(classes))
    print("Patient IDs in more than one class:", (classes > 1).sum())


def inspect_image(path):
    """Return (width, height, color_mode, is_ok) for one image file."""
    try:
        with Image.open(path) as image:
            image.load()
            return image.width, image.height, image.mode, True
    except (OSError, SyntaxError, ValueError):
        return 0, 0, "", False


def add_image_info(table, image_dir):
    """Return a copy of the table with size, color mode and readable flag."""
    info = [inspect_image(Path(image_dir) / path) for path in table["relpath"]]
    columns = ["width", "height", "color_mode", "is_ok"]
    return table.join(pd.DataFrame(info, columns=columns, index=table.index))


def find_duplicates(table):
    """Return one row for each group of byte-identical files."""
    repeated = table[table.duplicated("md5", keep=False)]
    return repeated.groupby("md5").agg(
        copies=("name", "size"),
        labels=("label", "nunique"),
        patients=("patient_id", "nunique"))


def print_quality_summary(table):
    """Print corrupted files, color modes, image sizes and duplicates."""
    readable = table[table["is_ok"]]
    print("Corrupted files:", (~table["is_ok"]).sum())
    print("Color modes:", readable["color_mode"].value_counts().to_dict())
    sizes = readable.groupby(["width", "height"]).size()
    print("Different image sizes:", len(sizes))
    print(sizes.sort_values(ascending=False).head(10))
    groups = find_duplicates(table)
    print("Duplicate groups:", len(groups))
    print("Extra copies:", (groups["copies"] - 1).sum())
    print("Groups with different labels:", (groups["labels"] > 1).sum())
    print("Groups with different patient IDs:", (groups["patients"] > 1).sum())


def clean_table(table):
    """Return the table without broken files, label conflicts or extra copies."""
    readable = table[table["is_ok"]]
    labels_per_hash = readable.groupby("md5")["label"].transform("nunique")
    consistent = readable[labels_per_hash == 1]
    cleaned = consistent.drop_duplicates("md5").reset_index(drop=True)
    print("Removed broken files:", len(table) - len(readable))
    print("Removed label conflicts:", len(readable) - len(consistent))
    print("Removed extra copies:", len(consistent) - len(cleaned))
    print("Images kept:", len(cleaned))
    return cleaned


def assign_splits(table, n_folds, val_folds, test_folds, seed):
    """Add a split column, keeping each patient ID in exactly one split."""
    splitter = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=seed)
    folds = splitter.split(table, table["label"], groups=table["patient_id"])
    result = table.assign(split="train")
    for number, (_, rows) in enumerate(folds):
        if number < test_folds:
            result.loc[result.index[rows], "split"] = "test"
        elif number < test_folds + val_folds:
            result.loc[result.index[rows], "split"] = "val"
    return result


def check_no_patient_overlap(table):
    """Stop with an error if a split is empty or two splits share a patient."""
    ids = {name: set(table.loc[table["split"] == name, "patient_id"])
           for name in ["train", "val", "test"]}
    assert all(ids.values()), "A split is empty"
    assert not ids["train"] & ids["val"], "Patients shared by train and val"
    assert not ids["train"] & ids["test"], "Patients shared by train and test"
    assert not ids["val"] & ids["test"], "Patients shared by val and test"


def print_split_summary(table):
    """Print images, class percentages and patients for each split."""
    order = ["train", "val", "test"]
    images = pd.crosstab(table["label"], table["split"])[order]
    print("Images per class:\n", images)
    print("Class % inside each split:\n",
          (images / images.sum() * 100).round(2))
    print("Split % of all images:\n",
          (images.sum() / len(table) * 100).round(2))
    print("Patients per split:\n",
          table.groupby("split")["patient_id"].nunique()[order])


def save_splits(table, path):
    """Save filepath, label, patient_id and split to a CSV file."""
    renamed = table.rename(columns={"relpath": "filepath"})
    columns = ["filepath", "label", "patient_id", "split"]
    renamed[columns].to_csv(path, index=False)


def load_splits(path):
    """Read splits.csv with text patient IDs and check for leakage again."""
    table = pd.read_csv(path, dtype={"patient_id": str})
    check_no_patient_overlap(table)
    return table


def load_image(path, image_size):
    """Read an image as a grayscale uint8 tensor (size, size, 1)."""
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, (image_size, image_size), antialias=True)
    image = tf.cast(tf.round(image), tf.uint8)
    image.set_shape((image_size, image_size, 1))
    return image


def prepare_image(image, mode):
    """Make 3 float channels and apply the scaling the model expects."""
    image = tf.image.grayscale_to_rgb(tf.cast(image, tf.float32))
    if mode == "none":
        return image
    if mode == "rescale":
        return image / 255.0
    if mode == "vgg16":
        return tf.keras.applications.vgg16.preprocess_input(image)
    if mode == "resnet50":
        return tf.keras.applications.resnet50.preprocess_input(image)
    raise ValueError(f"Unknown preprocessing mode: {mode}")


def preprocess_single_image(path, image_size, mode):
    """Turn one raw image file into a model-ready batch of one."""
    image = prepare_image(load_image(path, image_size), mode)
    return tf.expand_dims(image, axis=0)
