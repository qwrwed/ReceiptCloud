import os
from google.cloud import vision
from enum import Enum

from PIL import Image, ImageDraw
import argparse
import io

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
        break_types.EOL_SURE_SPACE: " <ESS>\n" if desc else "\n",
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
        draw.polygon([
            bound.vertices[0].x, bound.vertices[0].y,
            bound.vertices[1].x, bound.vertices[1].y,
            bound.vertices[2].x, bound.vertices[2].y,
            bound.vertices[3].x, bound.vertices[3].y], None, color)
    return image

def image_to_data(path, feature=FeatureType.WORD, desc_bt=False):
    """Detects text in the file."""

    client = vision.ImageAnnotatorClient()

    with io.open(path, "rb") as image_file:
        content = image_file.read()

    #print(type(content))
    #input()
    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)
    document = response.full_text_annotation

    entries = []
    text = ""
    
    # Collect features by enumerating all document features
    for page_num in range(len(document.pages)):
        page = document.pages[page_num]

        for block_num in range(len(page.blocks)):
            block = page.blocks[block_num]

            for paragraph_num in range(len(block.paragraphs)):
                paragraph = block.paragraphs[paragraph_num]

                for word_num in range(len(paragraph.words)):
                    word = paragraph.words[word_num]

                    for symbol_num in range(len(word.symbols)):
                        symbol = word.symbols[symbol_num]

                        if feature == FeatureType.SYMBOL:
                            entry = dict()
                            entry["text"] = symbol.text
                            entry["bounding_box"] = symbol.bounding_box
                            entries.append(entry)
                        else:
                            text += symbol.text
                            if symbol.property.detected_break:
                                text += breaktype_to_symbol(
                                    symbol.property.detected_break.type_,
                                    desc_bt
                                )

                    if feature == FeatureType.WORD:
                        entry = dict()
                        entry["text"] = text
                        entry["bounding_box"] = word.bounding_box
                        entries.append(entry)
                        text = ""

                if feature == FeatureType.PARA:
                    entry = dict()
                    entry["text"] = text
                    entry["bounding_box"] = paragraph.bounding_box
                    entries.append(entry)
                    text = ""

            if feature == FeatureType.BLOCK:
                entry = dict()
                entry["text"] = text
                entry["bounding_box"] = block.bounding_box
                entries.append(entry)
                text = ""

    verbose = False
    if verbose:
        for i in range(len(entries)):
            print(f"\n{str(feature)} No. {i}")
            #print(entries[i]["text"])
            print(entries[i])
            input()

    #     vertices = (['({},{})'.format(vertex.x, vertex.y)
    #                 for vertex in text.bounding_poly.vertices])


    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    return entries

def render_doc_text(filein, fileout=0):
    image = Image.open(filein)
    entries = detect_text(filein, FeatureType.BLOCK)
    draw_boxes(image, entries, 'blue')

    entries = detect_text(filein, FeatureType.PARA)
    draw_boxes(image, entries, 'red')

    entries = detect_text(filein, FeatureType.WORD)
    draw_boxes(image, entries, 'yellow')

    if fileout != 0:
        image.save(fileout)
    else:
        image.show()

def data_raw_to_fulltext(data_raw):
    fulltext = ''.join([entry["text"] for entry in data_raw])
    return fulltext

if __name__ == '__main__':
    #render_doc_text("receipt.jpg")
    data_raw = image_to_data("receipt.jpg", FeatureType.BLOCK)
    data_txt = data_raw_to_fulltext(data_raw)
    print(data_txt)
