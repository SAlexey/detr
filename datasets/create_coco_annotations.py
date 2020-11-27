
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


def main():
    root = Path("/scratch/visual/ashestak/oai/v00/numpy/full")
    # move the files: 

    train_path = root/"train"
    test_path = root/"test"

    for each in (train_path, test_path):
        print(f"PATH: {each}")
        annotation_dict = {
            "info": {
                "author": "Alexey Shestakov",
                "year": 2020,
                "description": ("""
                Detection Anotation File 
                Sagittal 3D Dess MRI Dataset
                Boxes are in form [z1, z2, x1, x2, y1, y2]
                """)
            },
            "images": [], 
            "annotations": [],
        }
        for path in tqdm(list(each.iterdir())):
            item = np.load(path)
            image = item.get("image")
            d, w, h = image.shape
            annotation_dict["images"].append({
                "id": int(path.stem),
                "width": w,
                "height": h,
                "depth": d,
                "side": item.get("side")
            })

            for box, label in zip(item["boxes"], item["labels"]):
                z1, z2, x1, x2, y1, y2 = box
                annotation_dict["annotations"].append({
                    "id": len(annotation_dict["annotations"]),
                    "image_id": int(path.stem),
                    "category_id": label,
                    "bbox": [z1, x1, y1, z2 - z1, x2 - x1, y2 - y1]
                })

        with open(each/"annotation3d.json", "w") as fp:
            json.dump(annotation_dict, fp)
        

if __name__ == "__main__":
    main()