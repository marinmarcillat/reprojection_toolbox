import fiftyone as fo
from object_detection import fifty_one_utils as fou
import fiftyone.utils.random as four
import os

images_path = r"E:\ATM\images"
export_dir = r"E:\ATM\yolo_training_dataset"
train_split = 0.8

# Ex: your custom label format


# Create samples for your data
samples = []
classes = []
for root, dirs, files in os.walk(images_path):
        for d in dirs:
            classes.append(d)
            for file in os.listdir(os.path.join(root, d)):
                sample = fo.Sample(filepath=os.path.join(root, d, file))
                sample["ground_truth"] = fo.Classification(label=d)
                samples.append(sample)


# Create dataset
fou.delete_all_datasets()
dataset = fo.Dataset("classification-dataset")
dataset.add_samples(samples)
dataset.save()

four.random_split(
    dataset,
    {"train": train_split, "val": 1 - train_split}
)
splits = ["train", "val"]

for split in splits:
    view = dataset.match_tags(split)
    split_dir = os.path.join(export_dir, split)
    view.export(
        export_dir=split_dir,
        dataset_type=fo.types.ImageClassificationDirectoryTree,
        label_field="ground_truth"
    )





