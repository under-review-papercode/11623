import json
import os


def rad_splitter():
    SOURCE_DIR = "ANONYMIZED/vqa-rad/VQA_RAD Dataset Public.json"
    DUMP_DIR = "/ANONYMIZED/cache/RAD_M2I2/"

    ds = json.load(open(SOURCE_DIR, "r"))
    test = [d for d in ds if "test" in d["phrase_type"]]
    train = [d for d in ds if "test" not in d["phrase_type"]]
    assert len(test) == 451, f"found {len(test)} != from what declared in the RAD paper"
    with open(os.path.join(DUMP_DIR, "train.json"), "w") as f:
        f.write(json.dumps(train))
    with open(os.path.join(DUMP_DIR, "test.json"), "w") as f:
        f.write(json.dumps(test))


if __name__ == '__main__':
    rad_splitter()
