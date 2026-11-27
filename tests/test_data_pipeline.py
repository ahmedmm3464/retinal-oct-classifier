import pytest
import pandas as pd
from src.data import parse_filename , inspect_image ,index_files , build_image_table ,clean_table , check_no_patient_overlap
from PIL import Image
def test_parse_filename_valid():
    label , patient_id = parse_filename("CNV-12345-1.jpeg")
    assert label == "CNV" 
    assert patient_id =="12345"
    
def test_inspect_image_valid(tmp_path):
    img_path = tmp_path / "ok.jpeg"
    Image.new("L", (64, 32)).save(img_path)
    width, height, color_mode, is_ok = inspect_image(img_path)
    assert (width, height,color_mode, is_ok)  == (64, 32, "L", True)
    
def test_inspect_image_corrupted(tmp_path):
    bad_path = tmp_path / "bad.jpeg"
    bad_path.write_bytes(b"not an image")
    width, height, color_mode, is_ok = inspect_image(bad_path)
    assert (width, height,color_mode, is_ok)  == (0, 0, "", False)

def test_index_files_valid(tmp_path):
    cnv_dir = tmp_path / "CNV"
    cnv_dir.mkdir()
    img = cnv_dir / "CNV-12345-1.jpeg"
    img.write_bytes(b"fake image")
    df = index_files(tmp_path)
    assert len(df) == 1 
    assert df.iloc[0]["relpath"] == "CNV/CNV-12345-1.jpeg"
    assert df.iloc[0]["name"] == "CNV-12345-1.jpeg"
    assert (len(df.iloc[0]["md5"]) == 32)   
    
def test_index_files_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        index_files(tmp_path / "missing_folder")

def test_build_image_table_valid(tmp_path) :
    cnv_dir = tmp_path / "CNV"
    cnv_dir.mkdir()
    (cnv_dir / "CNV-12345-1.jpeg").write_bytes(b"fake")
    (cnv_dir / "notes.txt").write_text("ignore")
    df = build_image_table(tmp_path)
    assert len(df) == 1 
    assert df.iloc[0]["label"] == "CNV"
    assert df.iloc[0]["patient_id"] == "12345"

def test_build_image_table_mismatch(tmp_path):
    cnv_dir = tmp_path/"CNV"
    cnv_dir.mkdir()
    (cnv_dir / "DME-12345-1.jpeg").write_bytes(b"fake")
    with pytest.raises(ValueError):
        build_image_table(tmp_path)

def test_clean_table():
    df = pd.DataFrame({
        "md5": ["h1", "h2", "h1", "h3", "h3"],
        "label": ["CNV", "DME", "CNV", "CNV", "NORMAL"],
        "is_ok": [True, False, True, True, True],
    })
    cleaned = clean_table(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["md5"] == "h1"
    
def test_check_no_patient_overlap_valid():
    df = pd.DataFrame({
        "patient_id": ["p1", "p2", "p3"],
        "split": ["train", "val", "test"],
    })
    check_no_patient_overlap(df)


def test_check_no_patient_overlap_leakage():
    df = pd.DataFrame({
        "patient_id": ["p1", "p1", "p3"],
        "split": ["train", "val", "test"],
    })
    with pytest.raises(AssertionError):
        check_no_patient_overlap(df)
            
def test_parse_filename_invalid():
    with pytest.raises(ValueError):
        parse_filename("CNV-12345")
        

    
        
