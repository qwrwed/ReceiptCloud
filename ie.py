def raw_to_entries(data_raw):
    entries = []
    for i in range(len(data_raw[list(data_raw.keys())[0]])):
        entry = {list_name: data_raw[list_name][i] for list_name in data_raw.keys()}
        if entry["text"].strip() != "":
            entry["right"] = entry["left"] + entry["width"]
            entry["bottom"] = entry["top"] + entry["height"]
            entries.append(entry)
    return entries


def entries_to_lines(entries, include_text=True, include_bbox=True):
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

        line = ",".join([str(x) for x in line]).upper()
        lines.append(line)
    return lines

def extract_key_info():
    info = {
        "company": "BOOK TA .K (TAMAN DAYA) SDN BHD",
        "date": "25/12/2018",
        "address": "NO.53 55,57 & 59, JALAN SAGU 18, TAMAN DAYA, 81100 JOHOR BAHRU, JOHOR.",
        "total": "9.00"
    }
    return info


def gt_to_data(filepath):
    entries = []
    keys = ["left", "top", "right", "_", "_", "bottom", "_", "_", "text"]
    with open(filepath, "r") as f:
        for line in f:
            values = line.split(",")
            entry = dict(zip(keys, values))
            entry = {k: (int(v) if k != "text" else v) for k, v in entry.items()}
            del entry["_"]
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
    return entries_out