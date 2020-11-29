
from collections import defaultdict
import os
from pathlib import Path
import json 
import numpy as np
from tqdm import tqdm

def default_ann():
    return {
        "info": {},
        "images": [],
        "annotations": [],
        "licences": []
    }

def create_annotations(root):
    
    is_valid = lambda path: path.suffix == ".npz"
    
    ret = defaultdict(lambda: {
        "images": [], 
        "annotations": [], 
        "categories": [{"id": 0, "name": "LM"}, {"id": 1, "name": "MM"}]
    })

    config = (
        ("sag", (1, 2), [2, 3, 4, 5]), 
        ("axi", (0, 2), [0, 1, 4, 5]), 
        ("cor", (0, 1), [0, 1, 2, 3])
    )

    for each in root.iterdir():
        if is_valid(each):
            item = np.load(each)
            inputs = item["image"]
            boxes = item["boxes"]
            labels = item["labels"]
            shape = inputs.shape 
            image_id = int(each.stem)

            
            for ax, (h, w), ind in config:
                bbox = boxes[:, ind]
                image = {
                    "id": image_id,
                    "width": shape[w],
                    "height": shape[h],
                }
                annos = [
                    {
                        "id": len(ret[ax]["annotations"]) + j,
                        "image_id": image_id,
                        "category_id": int(label),
                        "iscrowd": 0,
                        "area": (box[1] - box[0]) * (box[3] - box[2]),
                        "bbox": [box[0], box[2], box[1] - box[0], box[3] - box[2]]
                    }
                    for j, (label, box) in enumerate(zip(labels, bbox))
                ]
                ret[ax]["images"].append(image)
                ret[ax]["annotations"].extend(annos)

    with open(root / "annotations.json", "w") as fp:
        ret = dict(ret)
        ret["config"] = config
        print(f"writing annotations to {root / 'annotations.json'}")
        json.dump(ret, fp)
        print("done")
    return ret


def main():
    root = Path("/scratch/visual/ashestak/oai/v00/numpy/full")
    
    train_annotations = create_annotations(root/"train")
    test_annotations = create_annotations(root/"test")

if __name__ == "__main__":
    main()