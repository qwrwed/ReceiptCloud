import os

import requests
import dotenv
from dotenv import load_dotenv

load_dotenv()

from PIL import Image, ImageDraw
import argparse
import io
import re
import copy
import sys
import pprint
import cv2
import ip
import ocr

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join("key.json")

# https://cloud.google.com/vision/docs/fulltext-annotations



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


def naive_char_width(entry):
    print(entry)
    input()


def pppprint(d):
    pprint.PrettyPrinter(indent=4).pprint(d)


def render_doc_text(img_arr, fileout=0):
    # image = Image.open(filein)
    # image = img_bytes
    img_pil = ip.cvt_cv2_pil(img_arr)
    entries = ocr.cloud_image_to_data(img_arr, ocr.FeatureType.BLOCK)
    draw_boxes(img_pil, entries, "blue")

    entries = ocr.cloud_image_to_data(img_arr, ocr.FeatureType.PARA)
    draw_boxes(img_pil, entries, "red")

    entries = ocr.cloud_image_to_data(img_arr, ocr.FeatureType.WORD)
    draw_boxes(img_pil, entries, "yellow")

    if fileout != 0:
        img_pil.save(fileout)
    else:
        img_pil.show()


def data_lines_to_text_lines(lines):
    return [line["text"] for line in lines]


def data_raw_to_fulltext(data_raw):
    fulltext = "".join([entry["text"] for entry in data_raw])
    return fulltext


def any_in(list_queries, string_to_search):
    for q in list_queries:
        if q in string_to_search:
            return True
    return False

def pipeline_load(path):
    img_arr = cv2.imread(path)
    return pipeline(img_arr)


def pipeline(img_arr):
    
    # img_arr = ip.grayscale(img_arr)
    # img_arr, _ = ip.threshold_otsu(img_arr)
    # img_arr = ip.filter_bilateral(img_arr)
    # img_arr = ip.threshold_adaptive(img_arr)
    # cv2.imshow("", img_arr)
    # cv2.waitKey()

    # render_doc_text(img_arr)
    #data_lines = cloud_ocr(img_arr, feature_type="word")

    data_lines = ocr.ocr_google(img_arr)
    # data_lines = ocr.ocr_tesseract(img_arr)
    #pppprint(data_lines)

    
    # pppprint(data_raw)
    # data_txt = data_raw_to_fulltext(data_raw)
    
    text_lines = data_lines_to_text_lines(data_lines)
    #pppprint(text_lines)
    

    items = []
    buffer = ""
    # pattern for arbitrary text ending in price
    pattern = "(.+)\s£?([0-9]+\.[0-9]{2})$"
    for i in range(len(text_lines)):
        # replace misinterpreted asterisks with asterisks
        text_lines[i] = text_lines[i].replace(" X ", " * ")
        text_lines[i] = text_lines[i].replace(" x ", " * ")
        # remove asterisks
        text_lines[i] = text_lines[i].replace(" * ", " ")

        text_lines[i] = text_lines[i].replace("€", "£")
        # "£" may be misread

        buffer += text_lines[i]
        match = re.search(pattern, buffer)
        if match:
            item = dict()
            item["name"], item["price"] = match.groups()
            items.append(item)
            buffer = ""
        else:
            buffer += " "
    
    text_full = "\n".join([item["name"] for item in items])
    # text_full += "\nPERFUME\nCHOCOLATE"
    price_full = str(sum([float(item["price"]) for item in items]))
    #print([text_full, price_full])
    #items.insert(0, {"name": text_full, "price": price_full})
    #pppprint(items)

    print("\n==OCR RESULT==")
    print(text_full)

    url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
    headers = {
        "x-app-id": os.environ.get("NUTX_APP_ID"),
        "x-app-key": os.environ.get("NUTX_APP_KEY"),
        "x-remote-user-id": os.environ.get("NUTX_USR_ID"),
    }

    aggregate = False

    request_data = {
        "query": text_full,
        # "line_delimited": True,
        # "use_branded_foods": True,
    }
    if aggregate:
        request_data.update({"aggregate":"summary"})

    response = requests.post(url, data=request_data, headers=headers)
    response_json = response.json()
    #pppprint(response_json)

    info_quantities = dict()
    # non-foods/non-results are skipped, so it is not possible to map the queried names onto the response items
    if response.status_code != 200:
        return None, None
    item_list = response_json['foods']
    for i in range(len(item_list)):
        item = item_list[i]
        del item["full_nutrients"] # for readability; nutrient ids are indecipherable without GET to https://trackapi.nutritionix.com/v2/utils/nutrients anyway
        del item["alt_measures"] # also for readability
        for k,v in item.items():
            if k.startswith("nf_"):
                if item[k] == None:
                    item[k] = 0
        # pppprint(item)
        item["nf_carbohydrate_non_sugar"] = item["nf_total_carbohydrate"] - item["nf_sugars"]
        item["nf_fat_non_saturated"] = item["nf_total_fat"] - item["nf_saturated_fat"]
        for k, v in item.items():
            if k.startswith("nf_"):
                if k not in info_quantities:
                    info_quantities[k] = list()
                info_quantities[k].append(v)
    
    info_quantities_sums = {k: round(sum(v), 2) for k,v in info_quantities.items()}        
    info = {
        "summary": info_quantities_sums,
        "list": item_list
    }
    #item_list.insert(0, info_quantities_sums)
    print("\n==NUTX RESULT==")
    pppprint(info)

    #info_full = response_json['foods'][0]
    #del info_full["full_nutrients"] # for readability; nutrient ids are indecipherable without GET to https://trackapi.nutritionix.com/v2/utils/nutrients anyway


    


    # raise RuntimeError
    # return
    # # pppprint(text_lines)
    # #n_queries = len(items)
    # for i in range(n_queries):
    #     item = items[i]
    #     data = {"query": item["name"]}
    #     print(data)
    #     response = requests.post(url, data=data, headers=headers)
    #     response_json = response.json()
        

    #     pppprint(response_json)
    #     #input()
    #     item["summary"] = (i == 0)
    #     if response.status_code == 200:
    #         #pppprint(response_json["foods"])
    #         # response_json["foods"] always seems to have 1 value
    #         item["nutx_info"] = response_json["foods"][0]
    #         del item["nutx_info"]["full_nutrients"] # for readability; nutrient ids are indecipherable without GET to https://trackapi.nutritionix.com/v2/utils/nutrients anyway

    #         #print(f"{len(response_json['foods'])} items")
    #     else:
    #         #pppprint(response)
    #         #print(f"0 items")
    #         #print(response_json["message"])
    #         item["nutx_info"] = None
    #     #input()
    #     #print("\n"*100)
    #     #pppprint(response.keys())
    #     #pppprint(response)
        
    #     # #search_result = openfoodfacts.products.search(item["name"])
    #     # #https://github.com/Clement-O/Projet_05/blob/70cf42ea5f69fa8226622c3869d1faca61f9b9eb/products/api.py
    #     # search_result = openfoodfacts.products.advanced_search({
    #     #     "search_terms": item["name"],
    #     #     # insufficient documentation to create a query which requests only English results
    #     #     #"tagtype_0": "countries",
    #     #     #"tag_contains_0": "contains",
    #     #     #"tag_0": "uk",
    #     #     "page_size":1000,
    #     #     #"lang": "en",
    #     #     #"cc": "en",
    #     #     #"lc": "en",
    #     # })

    #     # n_results = search_result['count']
    #     # print(f"{n_results} results found for '{item['name']}'")
    #     # #if n_results > 0:
    #     #     #pppprint(search_result['products'][0])

    #     # for product in search_result['products']:
    #     #     willDisplay = False
    #     #     if product["lang"] == "en":
    #     #         print("LANG == EN")
    #     #     else:
    #     #         print("OTHER LANG")

    #     #     pppprint((product["product_name"], product))
    #     #     input()
    #     #     print("\n"*100)
    #     #     #for k, v in product.items():

    #     # #pppprint([(x["product_name"], {k:v for k,v in x.items() if any_in(["lang", "lc", "cc", "categories"], k)}) for x in search_result['products']])
    #     # input()

    # # for item in items:
    # #     pppprint(item)
    # #     print()

    # # raise RuntimeError

    img_pil = ip.cvt_cv2_pil(img_arr)

    # draw_boxes(img_pil, data_raw, "blue")
    # draw_boxes(img_pil, data_lines, "green")
    # img_arr = ip.cvt_pil_cv2(img_pil)

    return info, img_arr
    # img_pil.show()


if __name__ == "__main__":
    path_default = "seg/S01200HQU10E.JPG"
    if len(sys.argv) == 0:
        path = path_default
    else:
        path = sys.argv[1]
    pipeline_load(path)
