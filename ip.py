import cv2
import numpy as np
from PIL import Image


def grayscale(img):
    # printv("Converting image to grayscale...")
    if len(img.shape) == 3:
        if img.shape[2] == 1:
            return np.squeeze(img)
        if img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        # printv("Image is already grayscale")
        return img


def color(img):
    # printv("Converting image to color...")
    if len(img.shape) == 2:
        # img = np.stack((img,)*3, axis=-1)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        # printv("Image is already in color")
        return img


def filter_gauss(img, k=5, sigmaX=0):
    # printv(f"Applying Gaussian filter with {kernelSize=} and {sigmaX=}...")
    return cv2.GaussianBlur(img, (k, k), sigmaX)


def filter_median(img, k=5):
    # printv(f"Applying median filter with {k=}...")
    return cv2.medianBlur(img, k)


def filter_bilateral(img, d=5, sigma=75):
    # printv(f"Applying bilateral filter with {d=} and {sigma=}...")
    return cv2.bilateralFilter(img, d, sigma, sigma)


def threshold_simple(img, val, maxVal=255):
    if is_grayscale(img):
        return cv2.threshold(img, val, maxVal, cv2.THRESH_BINARY)[1]
    else:
        return color(cv2.threshold(grayscale(img), val, maxVal, cv2.THRESH_BINARY)[1])


def threshold_otsu(img, maxVal=255):
    # printv(f"Applying Otsu thresholding...")
    if is_grayscale(img):
        ret, img = cv2.threshold(img, 0, maxVal, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        img = grayscale(img)
        ret, img = cv2.threshold(img, 0, maxVal, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = color(img)
    return img, ret


def threshold_adaptive(img, k=3, C=5, method="gaussian", maxVal=255):
    methods = {
        "gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        "mean": cv2.ADAPTIVE_THRESH_MEAN_C,
    }
    # printv(f"Applying adaptive Gaussian thresholding with {k=} and {C=}...")
    if is_grayscale(img):
        return cv2.adaptiveThreshold(
            img, maxVal, methods[method], cv2.THRESH_BINARY, k, C
        )
    else:
        return color(
            cv2.adaptiveThreshold(
                grayscale(img), maxVal, methods[method], cv2.THRESH_BINARY, k, C
            )
        )


def highpass_dct(img):
    img = np.float32(img)
    freqs = cv2.dct(img)
    freqs[:2, :2] = 0
    img = cv2.idct(freqs)
    img = (img - img.min()) / (img.max() - img.min())
    img = np.uint8(img * 255)
    return img


def rescale(img, scale_x, scale_y=None):
    if scale_y == None:
        scale_y = scale_x
    if (scale_x, scale_y) == (1, 1):
        return img

    # printv(f"Rescaling image: width *= {scale_x:.2f}, height *= {scale_y:.2f}")
    width = img.shape[1]
    height = img.shape[0]

    width = width * scale_x
    height = height * scale_y

    width = round(width)
    height = round(height)

    return cv2.resize(img, (width, height))


def canny(img, thresh_low, thresh_high):
    return cv2.Canny(img, thresh_low, thresh_high)


def get_selem(k, mode="ellipse"):
    # create structuring element (kernel) for morphological operations (dilation, erosion...)
    if isinstance(mode, str):
        mode_dict = {
            "ellipse": cv2.MORPH_ELLIPSE,
            "rect": cv2.MORPH_RECT,
            "cross": cv2.MORPH_CROSS,
        }
    elif isinstance(mode, int):
        mode_dict = {0: cv2.MORPH_ELLIPSE, 1: cv2.MORPH_RECT, 2: cv2.MORPH_CROSS}
    return cv2.getStructuringElement(mode_dict[mode], (k, k))


def dilate(img, k=5, mode="ellipse", it=1):
    element = get_selem(k, mode)
    return cv2.dilate(img, element, iterations=it)


def erode(img, k=5, mode="ellipse", it=1):
    element = get_selem(k, mode)
    return cv2.erode(img, element, iterations=it)


def open(img, k=5, mode="ellipse"):
    return dilate(erode(img, k, mode), k, mode)


def close(img, k=5, mode="ellipse"):
    return erode(dilate(img, k, mode), k, mode)


def morphgrad(img, k=5, mode="ellipse"):
    return dilate(img, k, mode) - erode(img, k, mode)


def tophat(img, k=5, mode="ellipse"):
    return img - dilate(erode(img, k, mode), k, mode)


def blackhat(img, k=5, mode="ellipse"):
    return erode(dilate(img, k, mode), k, mode) - img


def fill_holes(img_in):
    # https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/

    # Copy the thresholded image.
    img_floodfill = img_in.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = img_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from points (0, 0) and (-1, 0)
    cv2.floodFill(img_floodfill, mask, (0, 0), 255)
    cv2.floodFill(img_floodfill, mask, (w - 1, 0), 255)
    # Invert floodfilled image
    img_floodfill_inv = cv2.bitwise_not(img_floodfill)

    # Combine the two images to get the foreground.
    img_out = img_in | img_floodfill_inv
    return img_out


def keep_largest(img_in):
    # https://stackoverflow.com/a/56591448
    # Find largest contour in intermediate image
    contours, _ = cv2.findContours(img_in, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours == []:
        return img_in
    contour_max = max(contours, key=cv2.contourArea)

    # Output
    img_out = np.zeros(img_in.shape, np.uint8)
    cv2.drawContours(img_out, [contour_max], -1, 255, cv2.FILLED)
    img_out = cv2.bitwise_and(img_in, img_out)
    return img_out


def houghP(img, minLineLength=50, maxLineGap=10, orient_threshold=45):
    # https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
    info = cv2.HoughLinesP(img, 1, np.pi / 180, 50, None, minLineLength, maxLineGap)
    linesP = []
    if info is not None:
        for i in range(0, len(info)):
            l = info[i][0]
            angle = np.arctan2(l[3] - l[1], l[2] - l[0]) * 180 / np.pi
            linesP.append(
                {
                    "posx": (l[0], l[2]),
                    "posy": (l[1], l[3]),
                    "start": (l[0], l[1]),
                    "end": (l[2], l[3]),
                    "mid": ((l[0] + l[2]) // 2, (l[1] + l[3]) // 2),
                    "angle": angle,
                    "length": np.sqrt((l[3] - l[1]) ** 2 + (l[2] - l[0]) ** 2),
                    "orient": "h" if abs(angle) <= orient_threshold else "v",
                }
            )
    return linesP


def draw_linesP(
    img, linesP, params=["angle", "length", "orient"], color=(0, 255, 0), thickness=3
):
    if linesP == None:
        return img
    units = {
        "angle": "d",
        "length": "px",
    }
    for l in linesP:
        all_text = []
        for param in params:
            if isinstance(l[param], float):
                text = f"{l[param]:.0f}"
            else:
                text = l[param]
            if param in units:
                text += units[param]
            all_text.append(text)

        cv2.line(img, l["start"], l["end"], color, thickness, cv2.LINE_AA)
        cv2.putText(
            img, ":".join(all_text), l["mid"], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255)
        )
    return img


def is_grayscale(img):
    return len(np.shape(img)) == 2


def draw_compare_boxes(img, data1, data2, color1=(255, 0, 0), color2=(0, 0, 255)):
    img = color(img)
    boxes_1 = np.zeros_like(img, dtype=np.uint8)
    boxes_2 = np.zeros_like(img, dtype=np.uint8)

    draw_data(boxes_1, data1, color_boxes=color1)
    draw_data(boxes_2, data2, color_boxes=color2)

    boxes_1_mask = threshold_simple(boxes_1, 0)
    boxes_2_mask = threshold_simple(boxes_2, 0)

    boxes_mask = cv2.bitwise_or(boxes_1_mask, boxes_2_mask)
    boxes_mask_inv = cv2.bitwise_not(boxes_mask)
    boxes = cv2.addWeighted(boxes_1, 1, boxes_2, 1, 0)
    img = cv2.bitwise_and(img, boxes_mask_inv)
    img += boxes
    return img


def draw_data(
    img,
    data_by_entry,
    draw_boxes=True,
    draw_text=False,
    color_boxes=None,
    color_text=None,
    offset_text=False,
):
    if data_by_entry == []:
        return img
    if len(img.shape) == 3:  # color
        color_boxes = (255, 0, 0) if color_boxes is None else color_boxes
        color_text = (0, 255, 0) if color_text is None else color_text
    else:
        color_boxes = (0, 0, 0)
        color_text = (0, 0, 0)

    for entry in data_by_entry:
        bbox_start = (entry["left"], entry["top"])
        bbox_end = (entry["right"], entry["bottom"])
        thickness = 2
        if draw_boxes:
            img = cv2.rectangle(img, bbox_start, bbox_end, color_boxes, thickness)
        if draw_text:
            if offset_text:
                img = cv2.putText(
                    img,
                    entry["text"],
                    bbox_start,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=color_text,
                    thickness=2,
                )
            else:
                img = cv2.putText(
                    img,
                    entry["text"],
                    (bbox_start[0], bbox_end[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    color=color_text,
                    thickness=2,
                )
    return img


def cvt_cv2_pil(img_arr):
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_arr)
    return img_pil


def cvt_pil_cv2(img_pil):
    im_arr = np.asarray(img_pil)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    return img_arr


def cvt_cv2_bytes(img_arr):
    is_success, img_buf_arr = cv2.imencode(".jpg", img_arr)
    img_bytes = img_buf_arr.tobytes()
    return img_bytes
