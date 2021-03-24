import os
from google.cloud import vision
from google.cloud.vision import AnnotateImageResponse
from enum import Enum

from PIL import Image, ImageDraw
import argparse
import io
import coordinatesHelper
import copy
import json
import sys
import pprint
import cv2
import ip, ie
import numpy as np

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join("key.json")

# https://cloud.google.com/vision/docs/fulltext-annotations


class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5


def breaktype_to_symbol(bt, desc=False):
    break_types = vision.TextAnnotation.DetectedBreak.BreakType
    breaks = {
        break_types.SPACE: " ",
        break_types.SURE_SPACE: " <SS> " if desc else " ",
        # break_types.EOL_SURE_SPACE: " <ESS>\n" if desc else "\n",
        break_types.EOL_SURE_SPACE: " <ESS> " if desc else "\n",
        break_types.LINE_BREAK: "\n",
    }
    if bt in breaks:
        return breaks[bt]
    return ""


def draw_boxes(image, entries, color):
    """Draw a border around the image using the hints in the vector list."""
    draw = ImageDraw.Draw(image)

    for entry in entries:
        bound = entry["bounding_box"]
        draw.polygon(
            [
                bound["vertices"][0]["x"],
                bound["vertices"][0]["y"],
                bound["vertices"][1]["x"],
                bound["vertices"][1]["y"],
                bound["vertices"][2]["x"],
                bound["vertices"][2]["y"],
                bound["vertices"][3]["x"],
                bound["vertices"][3]["y"],
            ],
            None,
            color,
        )
    return image


def get_merged_lines(lines, raw_text):

    merged_array = []
    while len(lines) != 1:
        l = lines.pop()
        l1 = l
        status = True

        data = ""
        merged_element = None

        while True:
            w_element = raw_text.pop()

            # if w_element == None:
            #     break

            w = w_element["description"]

            index = l.index(w)
            temp = None

            l = l[index + len(w) :]
            if status:
                status = False

                merged_element = w_element

            if l == "":
                merged_element["description"] = l1
                merged_element["bounding_poly"]["vertices"][1] = w_element[
                    "bounding_poly"
                ]["vertices"][1]
                merged_element["bounding_poly"]["vertices"][2] = w_element[
                    "bounding_poly"
                ]["vertices"][2]
                merged_array.append(merged_element)
                break

    return merged_array


def init_line_segmentation(data):

    Y_MAX = coordinatesHelper.get_y_max(data)
    data = coordinatesHelper.invert_axis(data, Y_MAX)

    # The first index refers to the auto identified words which belongs to a sings line
    lines = data["text_annotations"][0]["description"].split("\n")

    # gcp vision full text
    raw_text = copy.deepcopy(data["text_annotations"])

    lines.reverse()
    raw_text.reverse()

    raw_text.pop()

    merged_array = get_merged_lines(lines, raw_text)

    coordinatesHelper.get_bounding_polygon(merged_array)
    coordinatesHelper.combine_bounding_polygon(merged_array)

    return construct_line_with_bounding_polygon(merged_array)


def construct_line_with_bounding_polygon(merged_array):
    final_array = []

    for i in range(len(merged_array)):
        if not merged_array[i]["matched"]:
            if len(merged_array[i]["match"]) == 0:
                final_array.append(merged_array[i]["description"])
            else:
                final_array.append(arrange_words_in_order(merged_array, 1))
    return final_array


def arrange_words_in_order(merged_array, k):
    merged_line = ""
    word_array = []
    line = merged_array[k]["match"]

    for i in range(len(line)):
        index = line[i]["match_line_num"]
        matched_word_for_line = merged_array[index]["description"]

        main_x = merged_array[k]["bounding_poly"]["vertices"][0]["x"]
        compare_x = merged_array[index]["bounding_poly"]["vertices"][0]["x"]

        if compare_x > main_x:
            merged_line = merged_array[k]["description"] + " " + matched_word_for_line
        else:
            merged_line = matched_word_for_line + " " + merged_array[k]["description"]

    return merged_line


def image_to_data(img_arr, feature=FeatureType.WORD, desc_bt=False):
    """
    Detects text in the image
    """

    client = vision.ImageAnnotatorClient()

    # with io.open(path, "rb") as image_file:
    #     content = image_file.read()
    img_bytes = ip.cvt_cv2_bytes(img_arr)
    image = vision.Image(content=img_bytes)

    response = client.document_text_detection(image=image)
    print("Response received")
    # https://stackoverflow.com/a/65728119
    response_json = AnnotateImageResponse.to_json(
        response, preserving_proto_field_name=True
    )
    response = json.loads(response_json)
    # um = init_line_segmentation(response)

    if "error" in response and "message" in response["error"]:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    if "full_text_annotation" not in response:
        return []
    # exit()
    document = response["full_text_annotation"]
    # print(document['text'])

    # pprint.PrettyPrinter(indent=4).pprint(document)
    # input()

    entries = []
    text = ""

    # Collect features by enumerating all document features
    for page_num in range(len(document["pages"])):
        page = document["pages"][page_num]

        for block_num in range(len(page["blocks"])):
            block = page["blocks"][block_num]

            for paragraph_num in range(len(block["paragraphs"])):
                paragraph = block["paragraphs"][paragraph_num]

                for word_num in range(len(paragraph["words"])):
                    word = paragraph["words"][word_num]

                    for symbol_num in range(len(word["symbols"])):
                        symbol = word["symbols"][symbol_num]

                        if feature == FeatureType.SYMBOL:
                            entry = dict()
                            entry["text"] = symbol["text"]
                            entry["bounding_box"] = symbol["bounding_box"]
                            entries.append(entry)
                        else:
                            text += symbol["text"]
                            if (
                                "property" in symbol
                                and "detected_break" in symbol["property"]
                            ):
                                text += breaktype_to_symbol(
                                    symbol["property"]["detected_break"]["type_"],
                                    desc_bt,
                                )

                    if feature == FeatureType.WORD:
                        entry = dict()
                        entry["text"] = text
                        entry["bounding_box"] = word["bounding_box"]
                        entries.append(entry)
                        text = ""

                if feature == FeatureType.PARA:
                    entry = dict()
                    entry["text"] = text
                    entry["bounding_box"] = paragraph["bounding_box"]
                    entries.append(entry)
                    text = ""

            if feature == FeatureType.BLOCK:
                entry = dict()
                entry["text"] = text
                entry["bounding_box"] = block["bounding_box"]
                entries.append(entry)
                text = ""

    entries = add_bbox_info(entries)

    verbose = False
    if verbose:
        for i in range(len(entries)):
            print(f"\n{str(feature)} No. {i}")
            # print(entries[i]["text"])
            print(entries[i])
            input()

    #     vertices = (['({},{})'.format(vertex['x'], vertex['y'])
    #                 for vertex in text.bounding_poly['vertices']])

    # for
    return entries


def add_bbox_info(entries):
    # https://stackoverflow.com/a/54443282
    for i in range(len(entries)):
        bbox = entries[i]["bounding_box"]
        verts = bbox["vertices"]
        verts_x = [verts[i]["x"] for i in range(len(verts))]
        verts_y = [verts[i]["y"] for i in range(len(verts))]
        bbox["center"] = {
            "x": sum(verts_x) / len(verts_x),
            "y": sum(verts_y) / len(verts_y),
        }
        bbox["size"] = {
            "x": max(verts_x) - min(verts_x),
            "y": max(verts_y) - min(verts_y),
        }
    return entries


def naive_char_width(entry):
    print(entry)
    input()


def pppprint(d):
    pprint.PrettyPrinter(indent=4).pprint(d)


def render_doc_text(img_arr, fileout=0):
    # image = Image.open(filein)
    # image = img_bytes
    img_pil = ip.cvt_cv2_pil(img_arr)
    entries = image_to_data(img_arr, FeatureType.BLOCK)
    draw_boxes(img_pil, entries, "blue")

    entries = image_to_data(img_arr, FeatureType.PARA)
    draw_boxes(img_pil, entries, "red")

    entries = image_to_data(img_arr, FeatureType.WORD)
    draw_boxes(img_pil, entries, "yellow")

    if fileout != 0:
        img_pil.save(fileout)
    else:
        img_pil.show()


def sort_data_2d(entries, threshold_factor=0.5):
    entries.sort(key=lambda e: e["bounding_box"]["center"]["y"])
    # pppprint(entries)

    diffs = [0]
    for i in range(1, len(entries)):
        diffs.append(
            entries[i]["bounding_box"]["center"]["y"]
            - entries[i - 1]["bounding_box"]["center"]["y"]
        )

    lines = []
    line = []
    # newline threshold = bbox height * threshold_factor
    for i in range(len(entries)):
        # print(diffs[i], [entries[i]['text']])
        threshold = entries[i]["bounding_box"]["size"]["y"] * threshold_factor
        if diffs[i] > threshold:
            lines.append(line)
            line = []
        line.append(entries[i])
    lines.append(line)

    for i in range(len(lines)):
        lines[i].sort(key=lambda e: e["bounding_box"]["center"]["x"])

    return lines


def group_data_lines(lines):

    # pppprint(lines)

    lines_combined = list()

    for line in lines:
        # pppprint(line)
        # input()
        line_combined = {
            "bounding_box": {
                "vertices": [
                    {"x": np.inf, "y": np.inf},  # x_min, y_min
                    {"x": 0, "y": np.inf},  # x_max, y_min
                    {"x": 0, "y": 0},  # x_max, y_max
                    {"x": np.inf, "y": 0},  # x_min, y_max
                ]
            },
            "text": "",
        }
        for component in line:
            line_combined["text"] += component["text"].replace("\n", " ")

            comp_verts = component["bounding_box"]["vertices"]
            line_verts = line_combined["bounding_box"]["vertices"]
            line_verts[0]["x"] = min(line_verts[0]["x"], comp_verts[0]["x"])
            line_verts[0]["y"] = min(line_verts[0]["y"], comp_verts[0]["y"])

            line_verts[1]["x"] = max(line_verts[1]["x"], comp_verts[1]["x"])
            line_verts[1]["y"] = min(line_verts[1]["y"], comp_verts[1]["y"])

            line_verts[2]["x"] = max(line_verts[2]["x"], comp_verts[2]["x"])
            line_verts[2]["y"] = max(line_verts[2]["y"], comp_verts[2]["y"])

            line_verts[3]["x"] = min(line_verts[3]["x"], comp_verts[3]["x"])
            line_verts[3]["y"] = max(line_verts[3]["y"], comp_verts[3]["y"])

            # line_combined["bounding_box"]["vertices"]
        line_combined["text"] = line_combined["text"].strip()
        lines_combined.append(line_combined)
    lines_combined = add_bbox_info(lines_combined)

    return lines_combined


def data_lines_to_text_lines(lines):
    return [line["text"] for line in lines]


def data_raw_to_fulltext(data_raw):
    fulltext = "".join([entry["text"] for entry in data_raw])
    return fulltext


def pipeline(path):
    img_arr = cv2.imread(path)
    # img_arr = ip.grayscale(img_arr)
    # img_arr, _ = ip.threshold_otsu(img_arr)
    # img_arr = ip.filter_bilateral(img_arr)
    # img_arr = ip.threshold_adaptive(img_arr)
    # cv2.imshow("", img_arr)
    # cv2.waitKey()

    # render_doc_text(img_arr)
    data_raw = image_to_data(img_arr, FeatureType.WORD, desc_bt=False)

    # pppprint(data_raw)
    # data_txt = data_raw_to_fulltext(data_raw)
    data_sorted = sort_data_2d(data_raw)
    # pppprint(data_sorted)
    data_lines = group_data_lines(data_sorted)
    text_lines = data_lines_to_text_lines(data_lines)
    pppprint(text_lines)

    img_pil = ip.cvt_cv2_pil(img_arr)
    draw_boxes(img_pil, data_raw, "blue")
    draw_boxes(img_pil, data_lines, "green")
    img_pil.show()


if __name__ == "__main__":
    path_default = "seg/S01200HQU10E.JPG"
    if len(sys.argv) == 0:
        path = path_default
    else:
        path = sys.argv[1]
    pipeline(path)
