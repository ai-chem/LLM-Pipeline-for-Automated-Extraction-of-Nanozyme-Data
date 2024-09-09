keys = [
    "formula",
    "syngony",
    "length, nm",
    "width, nm",
    "depth, nm",
    "surface",
    "pol",
    "surf",
    "Mw(coat), g/mol",
    "Km, mM",
    "Vmax, mM/s",
    "ReactionType",
    "C min, sub1,mM",
    "C max, sub1,mM",
    "C(const),co-sub 1 , mM (ко-субстрат)",
    "Ccat(mg/mL)",
    "ph",
    "temp, °C",
    "reaction type",
    "activity",
    "C(const),co-sub 2 , mM (ко-субстрат 2)",
    "C max, sub2,mM",
    "C min, sub2,mM",
    "substrate",
    "co-substrate",
]


def merge_dictionaries(dict_list):
    merged_dict = {key: [] for key in keys}

    for d in dict_list:
        for key, value in d.items():
            if key in merged_dict:
                if isinstance(value, list):
                    merged_dict[key].extend(value)
                else:
                    merged_dict[key].append(value)

    merged_dict = {k: v for k, v in merged_dict.items() if v}

    return merged_dict


if __name__ == "__main__":
    dict1 = {"formula": "H2O", "length, nm": 10, "pol": ["a", "b"], "ph": 7}

    dict2 = {"formula": "NaCl", "width, nm": 20, "pol": ["c"], "ph": 8}

    dict3 = {"syngony": "cubic", "length, nm": 15, "pol": "d", "temp, °C": 25}

    dict_list = [dict1, dict2, dict3]
    merged_dict = merge_dictionaries(dict_list)

    import pprint

    pprint.pprint(merged_dict)
