import os
import numpy as np, cv2
from tqdm import tqdm
from zipfile import ZipFile
import ip
import pytesseract
from pytesseract import Output
import subprocess
import json
import argparse
import imutils
from shutil import copyfile, rmtree
import cloud
import pprint as pp
import ocr

def process_img(img):
    # img = ip.grayscale(img)
    # img = ip.highpass_dct(img)
    # img = ip.filter_bilateral(img)
    # img, _ = ip.threshold_otsu(img)
    # img = ip.threshold_adaptive(img)
    return img

def zip_7z(
    name_outfile,
    input_dir,
    output_dir,
    pattern="*.txt",
    delete_existing=True,
    verbose=False,
):
    path_outfile = os.path.join(".", output_dir, name_outfile)
    path_outfile_ext = f"{path_outfile}.zip"
    path_7zip = os.path.join("7z", "7za")
    cmd_7zip = [path_7zip, "a", "-tzip", path_outfile]
    if isinstance(pattern, str):
        cmd_7zip.append(os.path.join(".", input_dir, pattern))
    elif isinstance(pattern, list):
        cmd_7zip.extend(
            [os.path.join(".", input_dir, filename) for filename in pattern]
        )

    if delete_existing and os.path.isfile(path_outfile_ext):
        if verbose:
            print(f"Removing {path_outfile_ext}")
        os.remove(path_outfile_ext)

    ret = subprocess.check_output(cmd_7zip)
    if verbose:
        print(ret.decode())

    return ret

def evaluate_single(file_txt, input_dir, output_dir, submit_dir):
    submit_file = os.path.join(submit_dir, file_txt)
    gt_file = os.path.join(input_dir, file_txt)
    tmp_dir = os.path.join(submit_dir, "tmp")
    copyfile(submit_file, os.path.join(tmp_dir, file_txt))
    ret = zip_7z("submit", tmp_dir, tmp_dir, pattern=file_txt)
    copyfile(gt_file, os.path.join(tmp_dir, file_txt))
    ret = zip_7z("gt", tmp_dir, tmp_dir, pattern=file_txt)
    ret = subprocess.check_output(
        [
            "py",
            os.path.join(".", output_dir, "script.py"),
            f"-g={os.path.join('.', tmp_dir,'gt.zip')}",
            f"-s={os.path.join('.', tmp_dir,'submit.zip')}",
        ]
    )
    return ret.decode()

def tqdm_range(n, use_tqdm=True):
    if use_tqdm:
        return tqdm(range(n))
    return range(n)

def gt_to_data(filepath):
    entries = []
    keys = ["left", "top", "right", "_", "_", "bottom", "_", "_", "text"]
    with open(filepath, "r") as f:
        for line in f:
            values = line.split(",")
            entry = {
                "bounding_box": {"vertices": [{"x": int(values[i]), "y": int(values[i+1])} for i in range(0, 8, 2)]},
                "text": ",".join(values[8:])
            }
            # entry = dict(zip(keys, values))
            #entry = {k: (int(v) if k != "text" else v) for k, v in entry.items()}
            #del entry["_"]
            entries.append(entry)
    ocr.add_bbox_info(entries)
    return entries

def process_files(files_img, ocr_engine, show_imgs=False, batch=True):

    if ocr_engine == "t":
        OCR_FN = ocr.ocr_tesseract
    elif ocr_engine == "g":
        OCR_FN = ocr.ocr_google

    if show_imgs:
        tmp_dir = os.path.join(submit_dir, "tmp")
        if os.path.exists(tmp_dir):
            rmtree(tmp_dir)
        os.makedirs(tmp_dir)

    include_bbox = task in (0, 1, 3)
    include_text = task in (0, 2, 3)
    task_1_group = ocr.FeatureType.LINE
    # task_1_group = ocr.FeatureType.LINE_SPLIT
    # task_1_group = ocr.FeatureType.WORD
    
    data_group = task_1_group if task in (0, 1) else ocr.FeatureType.WORD
    print("Group text by: ", end="")
    print(data_group)

    files_txt_new = set()
    for i in tqdm_range(len(files_img), not show_imgs):
        file_img = files_img[i]
        file_split = file_img.split(".")
        file_name = ".".join(file_split[:-1])
        file_ex = file_split[-1]
        file_txt = file_name + ".txt"

        img = cv2.imread(os.path.join(input_dir, file_img))
        img = process_img(img)

        

        data_raw = OCR_FN(img, data_group)
        data_SROIE = ocr.to_SROIE(data_raw, include_text, include_bbox, to_string=True)
    

        # pp.pprint(data_SROIE)
        # exit()

        if task in (0, 1, 2):
            txt_submit = "\n".join(data_SROIE)
        elif task == 3:
            print("Task 3 Not Supported")
            exit()
            # key_info = ie.extract_key_info()
            # txt_submit = json.dumps(key_info)

        # print([txt_submit])

        if batch:
            submit_file = os.path.join(submit_dir, file_txt)
            with open(submit_file, "w", encoding='utf-8') as f:
                f.write(txt_submit)
            files_txt_new.add(file_txt)

        if show_imgs:
            data_entries = data_raw
            # print(data_entries)
            # print(data_raw)
            data_gt = gt_to_data(os.path.join(input_dir, file_txt))
            img = ip.draw_compare_boxes(img, data_gt, data_entries)
            img = ip.draw_data(
                img,
                data_raw,
                draw_boxes=False,
                draw_text=False,
                color_text=(0, 127, 0),
                offset_text=True,
            )
            print(f"\nEvaluating {file_img}")
            result = evaluate_single(file_txt, input_dir, output_dir, submit_dir)
            print(result)
            cv2.imshow(file_img, img)
            cv2.waitKey()
            cv2.destroyAllWindows()

    if batch:
        files_txt_all = {
            file for file in os.listdir(submit_dir) if file.endswith(".txt")
        }
        files_txt_old = files_txt_all - files_txt_new
        for file in files_txt_old:
            os.remove(os.path.join(submit_dir, file))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SROIE task 1 or 2")
    parser.add_argument("-t", type=int, help="Task [1..3]")
    parser.add_argument("-n", type=int, help="Max number of files", default=None)
    parser.add_argument("-s", help="Show images", action="store_true")
    parser.add_argument("-d", help="Disable batch processing", action="store_true")
    parser.add_argument("-o", help="OCR engine [t for Tesseract, g for Google Cloud]", choices=['t', 'g'])
    args = parser.parse_args()
    task = args.t
    max_files = args.n
    batch = not args.d
    if not batch and not args.s:
        print(
            "\nBatch processing and individual processing disabled; remove -d flag or add -s flag for results"
        )
        print("Exiting...\n")
        exit()

    ds = "test"

    if task in (0, 1, 2):
        input_dir = os.path.join("SROIE", f"task1_2-{ds}")
    elif task == 3:
        input_dir = os.path.join("SROIE", f"task3-{ds}")
    output_dir = os.path.join("SROIE", "Evaluation Scripts", f"script_ch13_t{task}_e1")
    submit_dir = os.path.join(output_dir, "submit")

    for dir in (output_dir, submit_dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    files_img = [file for file in os.listdir(input_dir) if file.endswith("jpg")]
    files_gt = [file for file in os.listdir(input_dir) if file.endswith("txt")]
    if max_files:
        num_files = min(max_files, len(files_img))
        files_img = files_img[:num_files]
        files_gt = files_gt[:num_files]

    process_files(files_img, ocr_engine=args.o, show_imgs=args.s, batch=batch)

    if batch:
        verbose = False
        ret = zip_7z("gt", input_dir, output_dir, pattern=files_gt, verbose=verbose)
        ret = zip_7z("submit", submit_dir, output_dir, verbose=verbose)

        ret = subprocess.check_output(
            [
                "py",
                os.path.join(".", output_dir, "script.py"),
                f"-g={os.path.join('.', output_dir,'gt.zip')}",
                f"-s={os.path.join('.', output_dir,'submit.zip')}",
            ]
        )
        print("\nDisplaying overall results:")
        print(ret.decode())
