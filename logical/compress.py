def __compact(class_dict):
    color = []

    for cls, cnt in class_dict.items():
        s = ""
        if cnt > 1:
            s += str(cnt)
        s += cls

        color.append(s)

    color = ",".join(color)
    if len(color) > 1:
        color = "(" + color + ")"

    return color


def compress_by_feature(bind):
    comp = []
    feat_prev = -1
    class_dict = {}

    for b in bind:
        feat, cls = b["feat"], b["class"]

        if feat == feat_prev:
            if cls not in class_dict:
                class_dict[cls] = 0
            class_dict[cls] += 1
        else:
            class_dict = {cls: 1}
            feat_prev = feat

            comp.append({
                "feat": feat,
                "class": class_dict
            })

    return comp


def compress_by_class(bind):
    bind = compress_by_feature(bind)

    comp = [{
        "class": bind[0]["class"],
        "feat_from": bind[0]["feat"],
        "feat_to": bind[0]["feat"]
    }]

    for i in range(1, len(bind)):
        feat, cls = bind[i]["feat"], bind[i]["class"]
        prev = comp[-1]

        if len(prev["class"].keys()) == 1 and len(cls.keys()) == 1 and list(prev["class"].keys())[0] == list(cls.keys())[0]:
            color = list(cls.keys())[0]
            count = list(cls.values())[0]
            prev["class"][color] += count
            prev["feat_to"] = feat
        else:
            comp.append({
                "class": cls,
                "feat_from": feat,
                "feat_to": feat
            })

    return comp


def compress_to_class_str(comp):
    return " ".join(__compact(c["class"]) for c in comp)


def distribute(comp):
    comp[0]["feat_from"] = -float('inf')
    comp[len(comp) - 1]["feat_to"] = float('inf')

    for i in range(1, len(comp)):
        prev = comp[i - 1]
        cur = comp[i]

        mid_value = (prev["feat_to"] + cur["feat_from"]) / 2
        prev["feat_to"] = mid_value
        cur["feat_from"] = mid_value

    return comp
