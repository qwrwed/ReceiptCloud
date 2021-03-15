#https://github.com/sshniro/line-segmentation-algorithm-to-gcp-vision/blob/master/nodejs/coordinatesHelper.js
import copy
import matplotlib.path as pltPath

def get_y_max(data):
    """
    computes the maximum y coordinate from the identified text blob
    """
    v = data['text_annotations'][0]['bounding_poly']['vertices']
    y_array = []
    for i in range(4):
        y_array.append(v[i]['y'])
    return max(y_array)

def invert_axis(data, y_max):
    """
    inverts the y axis coordinates for easier computation
    as the google vision starts the y axis from the bottom
    """
    #data = fill_missing_values(data) # unnecessary with python vision 2.0: https://stackoverflow.com/a/65728119
    for i in range(1, len(data['text_annotations'])):
        v = data['text_annotations'][0]['bounding_poly']['vertices']
        for j in range(4):
            v[j]['y'] = y_max - v[j]['y']
    return data

def get_bounding_polygon(merged_array):
    for i in range(len(merged_array)):
        arr = []

        # calculate line height
        h1 = merged_array[i]['bounding_poly']['vertices'][0]['y'] - merged_array[i]['bounding_poly']['vertices'][3]['y']
        h2 = merged_array[i]['bounding_poly']['vertices'][1]['y'] - merged_array[i]['bounding_poly']['vertices'][2]['y']
        h = h1
        if h2 > h1:
            h = h2

        avg_height = round(h*0.6)

        arr.append(merged_array[i]['bounding_poly']['vertices'][1])
        arr.append(merged_array[i]['bounding_poly']['vertices'][0])

        line1 = get_rectangle(copy.copy(arr), True, avg_height, True)

        arr = []
        arr.append(merged_array[i]['bounding_poly']['vertices'][2])
        arr.append(merged_array[i]['bounding_poly']['vertices'][3])
        line2 = get_rectangle(copy.copy(arr), True, avg_height, True)

        merged_array[i]['bigbb'] = create_rect_coordinates(line1, line2)
        merged_array[i]['lineNum'] = i
        merged_array[i]['match'] = []
        merged_array[i]['matched'] = False
    
def combine_bounding_polygon(merged_array):
    # select one word from the array
    for i in range(len(merged_array)):

        bigbb = merged_array[i]['bigbb']

        # iterate through all the array to find the match
        for k in range(i, len(merged_array)):
            if (k != i) and (merged_array[k]['matched'] == False):
                inside_count = 0
                for j in range(4):
                    coordinate = merged_array[k]['bounding_poly']['vertices'][j]
                    if inside([coordinate['x'], coordinate['y']], bigbb):
                        inside_count += 1
                
                # all four points were inside the big bb
                if inside_count == 4:
                    match = {'match_count': inside_count, 'match_line_num': k}
                    merged_array[i]['match'].append(match)
                    merged_array[k]['matched'] = True

def inside(coord, bb):
    path = pltPath.Path(bb)
    return path.contains_points([coord])

def get_rectangle(v, is_round_values, avg_height, is_add):
    if is_add:
        v[1]['y'] += avg_height
        v[0]['y'] += avg_height
    else:
        v[1]['y'] -= avg_height
        v[0]['y'] -= avg_height

    y_diff = v[1]['y'] - v[0]['y']
    x_diff = v[1]['x'] - v[0]['x']

    gradient = y_diff / x_diff

    x_thresh_min = 1
    x_thresh_max = 2000

    y_min = None
    y_max = None
    if gradient == 0:
        y_min = v[0]['y']
        y_max = v[0]['y']
    else:
        y_min = (v[0]['y']) - (gradient * (v[0]['x'] - x_thresh_min))
        y_max = (v[0]['y']) + (gradient * (x_thresh_max - v[0]['x']))
    
    if is_round_values:
        y_min = round(y_min)
        y_max = round(y_max)
    
        return {'x_min' : x_thresh_min, 'x_max' : x_thresh_max, 'y_min': y_min, 'y_max': y_max}

def create_rect_coordinates(line1, line2):
        return [[line1['x_min'], line1['y_min']], [line1['x_max'], line1['y_max']], [line2['x_max'], line2['y_max']],[line2['x_min'], line2['y_min']]]