from datasets.oai import subjects_from_dicom
import numpy as np 
import shutil

def main():

    subjects = subjects_from_dicom(num_workers=5)
    sample = np.random.choice(subjects, 64)
    
    for each in sample:
        image = each["image"]
        label = each["label_map"]
        shutil.copytree(image.path, f"./sample/images/{image.path}")
        shutil.copytree(label.path, f"./sample/masks/{label.path}")


if __name__ == "__main__":
    main()