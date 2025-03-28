import geopandas as gpd
import pandas as pd
from Biigle.biigle import Api
from biigle_reclassifier.utils import get_tree
from tqdm import tqdm
from datetime import datetime, timedelta

navigation_path = r"D:\canyon_scale\data\2021\navigation\nav_ref_pl201-215_2021.shp"

label_tree_id = 60
label_id = 4929
label_name = "Habitats_CoralFish"

project_id = 26

navigation = gpd.read_file(navigation_path)
navigation['datetime'] = pd.to_datetime(navigation['date'].astype('str') + ' ' + navigation['time'].astype('str'), format = "mixed", dayfirst = True)
navigation = navigation.set_index(['datetime'])
navigation.sort_index(inplace=True, ascending=True)

api = Api()

if api is None:
    print("API connection failed")
    quit()

label_tree = api.get(
    f'label-trees/{label_tree_id}'
).json()["labels"]


lt = label_tree.copy()
tree= get_tree(lt, False)

sub_t = tree.subtree(label_id)

label_dict = {label_id: label_name}
for node in sub_t.all_nodes():
    label_dict[node.identifier] = node.tag

coral_fish_annotations = []
for l_id in tqdm(label_dict, total=len(label_dict)):
    annotations = api.get(
        f'projects/{project_id}/video-annotations/filter/label/{l_id}'
    ).json()
    if len(annotations) == 0:
        continue
    for ann_id in annotations:
        annotation = api.get(f'video-annotations/{ann_id}').json()
        if annotation["shape_id"] == 7:
            annotation["video_filename"] = api.get(f"videos/{annotation['video_id']}").json()["filename"]
            annotation["video_start_time"] = datetime.strptime(annotation["video_filename"].split("_")[-2],'%y%m%d%H%M%S')
            annotation["start_time"] = annotation["video_start_time"] + timedelta(seconds = annotation["frames"][0])
            annotation["end_time"] = annotation["video_start_time"] + timedelta(seconds = annotation["frames"][-1])
            coral_fish_annotations.append(annotation)

print("Saving results to nav file")

for annotation in tqdm(coral_fish_annotations, total=len(coral_fish_annotations)):
    nav_len = len(navigation.loc[annotation['start_time']:annotation['end_time']])
    if nav_len == 0:
        print(f"Annotation not in navigation: {annotation['video_filename']} {annotation['start_time']} {annotation['end_time']}")
    else:
        navigation.loc[annotation['start_time']:annotation['end_time'], "label"] = annotation['labels'][0]['label']['name']

navigation_with_labels = navigation[~navigation['label'].isnull()]

navigation_with_labels.to_file(r"D:\canyon_scale\data\2021\annotations\habitat_coralfish.shp")

print("stop")

    


