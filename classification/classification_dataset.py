import fiftyone as fo
from object_detection import fifty_one_utils as fou
import fiftyone.utils.random as four
from ultralytics import YOLO
from fiftyone import ViewField as F
import os

images_path = r"E:\ATM\images"
export_dir = r"E:\ATM\yolo_training_dataset"
model_path = r"E:\ATM\models\train4\weights\best.pt"
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

dataview = dataset.match_tags("val")

model = YOLO(model_path)

dataview.apply_model(model, label_field="classifications")
dataview.default_classes = list(model.names.values())
dataview.save()

result = dataview.evaluate_classifications("classifications", gt_field="ground_truth")

result.print_report()

print("Result")
print(f"Dataset val length: {dataview.count()}")
print(f"False predictions: {dataview.match(F('classifications.label') != F('ground_truth.label')).count()}")

session = fo.launch_app(dataview)
session.wait()



"""four.random_split(
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
    )"""





