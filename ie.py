
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
