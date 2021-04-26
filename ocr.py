# default ocr format should be Google-style, a list of the following:
    #  {   'bounding_box': {   'center': {'x': 201.5, 'y': 939.5},
    #                         'normalized_vertices': [],
    #                         'size': {'x': 249, 'y': 69},
    #                         'vertices': [   {'x': 78, 'y': 905},
    #                                         {'x': 326, 'y': 908},
    #                                         {'x': 325, 'y': 974},
    #                                         {'x': 77, 'y': 971}]},
    #     'text': 'REDUCED '},

from enum import Enum
from google.cloud import vision
from google.cloud.vision import AnnotateImageResponse
import json
import numpy as np
import pprint as pp
import pytesseract
from pytesseract import Output

import ie, ip

# UTILITIES

# tesseract default feature types:
#   'page_num', 'block_num', 'par_num', 'line_num', 'word_num'
#   (no symbol)
# google cloud default feature types:
#   'pages', 'blocks', 'paragraphs', 'words', 'symbols'
#   (no line)

class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PAR = 3
    LINE = 4
    WORD = 5
    SYMBOL = 6

    
    
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

def to_SROIE(entries, include_text=True, include_bbox=True, to_string=False):
    # convert from:
    # [
    #     {
    #         "bounding_box":
    #             {
    #                 "vertices": [
    #                     {"x": x1, y: y1}
    #                     ...
    #                     {"x": x4, y: y4}
    #                 ],
    #                 ...
    #             }
    #         "text": text
    #     }
    # ]
    #
    # to:
    # [
    #     [x1, y1, x2, y2, x3, y3, x4, y4, text],
    #     ...
    # ]
    lines = []
    for entry in entries:
        line = list()
        if include_bbox:
            for i in range(4):
                for d in ("x", "y"):
                    line.append(entry["bounding_box"]["vertices"][i][d])

        if include_text:
            line.append(entry["text"])

        if to_string:
            line = ",".join([str(x) for x in line]).upper()
        lines.append(line)
    return lines

def from_SROIE(entries):
    # convert from:
    # [
    #     [x1, y1, x2, y2, x3, y3, x4, y4, text],
    #     ...
    # ]
    #
    # to:
    # [
    #     {
    #         "bounding_box":
    #             {
    #                 "vertices": [
    #                     {"x": x1, y: y1}
    #                     ...
    #                     {"x": x4, y: y4}
    #                 ],
    #                 ...
    #             }
    #         "text": text
    #     }
    # ]
    lines = []
    for entry in entries:
        x1, y1, x2, y2, x3, y3, x4, y4, text = entry
        line = {
            "bounding_box": {
                "vertices": [
                    {"x": x1, "y": y1},
                    {"x": x2, "y": y2},
                    {"x": x3, "y": y3},
                    {"x": x4, "y": y4},
                ]
            },
            "text": text
        }
        lines.append(line)
    lines = add_bbox_info(lines)
    return lines

# TESSERACT OCR
def ocr_tesseract(img, feature_type=FeatureType.LINE):

    # optinally preprocess image
    # img = process(img)

    # pt_config=f"-c tessedit_write_images=true"
    # pt_config=f"-l=eng"##
    img = ip.grayscale(img)

    feature_type = feature_type.name.lower()
    
    pt_config = f""

    data_raw = pytesseract.image_to_data(img, output_type=Output.DICT, config=pt_config)
    # print("\n\n\n\n\n\nRAW DATA:")
    # print(data_raw)

    data_entries = raw_to_entries(data_raw)
    # print("\n\n\n\n\n\nDATA ENTRIES:")
    # pp.pprint(data_entries)

    data_entries = group_data(data_entries, feature_type)
    # print("\n\n\n\n\n\nDATA ENTRIES (grouped):")
    # pp.pprint(data_entries)
    
    data_lines = from_SROIE(entries_to_lines(data_entries))
    # print("\n\n\n\n\n\nDATA LINES:")
    # pp.pprint(data_lines)

    return data_lines

def raw_to_entries(data_raw):
    entries = []
    for i in range(len(data_raw[list(data_raw.keys())[0]])):
        entry = {list_name: data_raw[list_name][i] for list_name in data_raw.keys()}
        if entry["text"].strip() != "":
            entry["right"] = entry["left"] + entry["width"]
            entry["bottom"] = entry["top"] + entry["height"]
            entries.append(entry)
    return entries


def group_data(entries_in, key):
    key += "_num"
    entries_out = []
    prefix_names = []
    if len(entries_in) == 0:
        return []
    for x in entries_in[0].keys():
        if "_num" in x:
            if x == key:
                break
            prefix_names.append(x)
    group_num_prev = None
    prefix_prev = None
    group_entry = dict()

    for entry in entries_in:
        prefix_curr = {k: entry[k] for k in prefix_names}
        group_num_curr = entry[key]
        if (prefix_curr != prefix_prev) or (
            group_num_curr != group_num_prev
        ):  # different "line"
            if len(group_entry) > 0:
                group_entry["width"] = group_entry["right"] - group_entry["left"]
                group_entry["height"] = group_entry["bottom"] - group_entry["top"]
                entries_out.append(group_entry)
            group_entry = entry
        else:  # same "line"
            group_entry["text"] += " " + entry["text"]
            group_entry["left"] = min(group_entry["left"], entry["left"])
            group_entry["top"] = min(group_entry["top"], entry["top"])
            group_entry["right"] = max(group_entry["right"], entry["right"])
            group_entry["bottom"] = max(group_entry["bottom"], entry["bottom"])
        prefix_prev = prefix_curr
        group_num_prev = group_num_curr
    entries_out.append(group_entry)

    
    return entries_out


def entries_to_lines(entries, include_text=True, include_bbox=True, to_string=False):
    lines = []
    for entry in entries:
        line = list()
        if include_bbox:
            x1 = x4 = entry["left"]
            x2 = x3 = entry["right"]
            y1 = y2 = entry["top"]
            y3 = y4 = entry["bottom"]
            line.extend([x1, y1, x2, y2, x3, y3, x4, y4])

        if include_text:
            transcript = entry["text"]
            line.append(transcript)


        if to_string:
            line = ",".join([str(x) for x in line]).upper()
        lines.append(line)
    return lines


# GOOGLE CLOUD VISION OCR

def ocr_google(img_arr, feature_type=FeatureType.LINE):
    use_lines = False
    if feature_type == FeatureType.LINE:
        use_lines = True
        feature_type = FeatureType.WORD

    img_arr = ip.grayscale(img_arr)

    data = image_to_data(img_arr, feature_type, desc_bt=False)
    # pp.pprint(data)
    if use_lines:
        data = sort_data_2d(data)
        data = group_data_lines(data)

    strip_newlines = True
    if strip_newlines:
        for entry in data:
            entry["text"] = entry["text"].strip()

    return data

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
    # print("Response received")
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
    
    # pp.pprint(document)
    # print(document['text'])

    # pprint.PrettyPrinter(indent=4).pprint(document)
    # input()

    entries = []
    text = ""

    break_on_break = True # only treat as new word when break (" ", "\n") is detected

    entry = ocr_entry_init()

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
                        detected_break = None

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
                                detected_break = symbol["property"]["detected_break"]["type_"]

                    if feature == (FeatureType.WORD or FeatureType.LINE):
                        
                        if detected_break or not break_on_break: # the end of the word has been reached
                            #entry = ocr_entry_init()
                            entry["text"] = text
                            bbox_extend(entry["bounding_box"], word["bounding_box"])

                            # print("\n\nADDING WORD")
                            # pp.pprint(entry)

                            entries.append(entry)
                            
                            entry = ocr_entry_init()
                            text = ""
                        else: # this is not the full word, there is still more to add
                            # print("\n\nEXTENDING WORD")
                            # pp.pprint(entry)
                            # print("\n\nWITH")
                            # pp.pprint(word)

                            bbox_extend(entry["bounding_box"], word["bounding_box"])

                            # print("\n\nRESULT:")
                            # pp.pprint(entry)
                            # print("\n\n\n\n")

                if feature == FeatureType.PAR:
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

def bbox_extend(bbox_curr, bbox_new):
    line_verts = bbox_curr["vertices"]
    comp_verts = bbox_new["vertices"]
    line_verts[0]["x"] = min(line_verts[0]["x"], comp_verts[0]["x"])
    line_verts[0]["y"] = min(line_verts[0]["y"], comp_verts[0]["y"])

    line_verts[1]["x"] = max(line_verts[1]["x"], comp_verts[1]["x"])
    line_verts[1]["y"] = min(line_verts[1]["y"], comp_verts[1]["y"])

    line_verts[2]["x"] = max(line_verts[2]["x"], comp_verts[2]["x"])
    line_verts[2]["y"] = max(line_verts[2]["y"], comp_verts[2]["y"])

    line_verts[3]["x"] = min(line_verts[3]["x"], comp_verts[3]["x"])
    line_verts[3]["y"] = max(line_verts[3]["y"], comp_verts[3]["y"])

def ocr_entry_init():
    return {
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

def group_data_lines(lines):

    # pppprint(lines)

    lines_combined = list()

    for line in lines:
        # pppprint(line)
        # input()
        line_combined = ocr_entry_init()
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
