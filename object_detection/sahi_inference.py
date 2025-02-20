from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import fiftyone as fo
import object_detection.fifty_one_utils as fou
import pandas as pd

def predict_with_slicing(sample, label_field, detection_model, **kwargs):
    result = get_sliced_prediction(
        sample.filepath, detection_model, verbose=0, **kwargs
    )
    sample[label_field] = fo.Detections(detections=result.to_fiftyone_detections())

def sahi_inference(model_path, dataset, label_field = "detections", slice = 2000):

    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=0.25, ## same as the default value for our base model
        device="cuda", # or 'cuda'
    )

    kwargs = {"overlap_height_ratio": 0.2, "overlap_width_ratio": 0.2}

    print("SAHI inference")
    for sample in dataset.iter_samples(progress=True, autosave=True):
        predict_with_slicing(sample, label_field=label_field, detection_model=detection_model , slice_height=slice, slice_width=slice, **kwargs)
    dataset.default_classes = list(detection_model.model.names.values())
    dataset.save()
    return dataset

if __name__ == '__main__':
    model_path = r"D:\model_training\trained_models\coco_multilabel_yolov11l_datarmor\sliced_yolo_training\best.pt"
    dataset_dir = r"D:\model_training\trained_models\coco_multilabel_yolov11l_datarmor\coco_less_labels"

    dataset = fou.import_yolov5_format(dataset_dir, label_field = "ground_truth")

    dataset_inf = sahi_inference(model_path, dataset)

    result = fou.fo_to_csv(dataset_inf)

    pd.DataFrame(result).to_csv(r"D:\chereef_marin\chereef23\annotations\inference_multilabel_SAHI.csv", index = False)

