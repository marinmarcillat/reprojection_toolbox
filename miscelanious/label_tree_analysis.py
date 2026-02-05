from treelib import Node, Tree
from Biigle.biigle import Api
import pandas as pd
import json

api = Api()
lt_id = 60
export_file = r"D:/00_Local_scale/common/label_tree/label_tree_Lampaul_all.csv"

volume_id_list = [334,462]

annotations_labels_dict = {}
for vid in volume_id_list:
    annotations = api.get(f'volumes/{vid}/statistics').json()['annotationLabels']
    for al in annotations:
        if al['id'] in annotations_labels_dict:
            annotations_labels_dict[al['id']] += al["count"]
        else:
            annotations_labels_dict[al['id']] = al["count"]

lt = api.get(f'label-trees/{lt_id}').json()["labels"]

tree = Tree()
tree.create_node("Root", "root")
while len(lt) > 0:
    for l in lt:
        if l["parent_id"] is None:
            tree.create_node(l["name"], l["id"], parent="root")
            lt.remove(l)
        else:
            if tree.contains(l["parent_id"]):
                nb = 0
                if l["id"] in annotations_labels_dict:
                    nb = annotations_labels_dict[l["id"]]
                tree.create_node(l['name'], l["id"], parent=l["parent_id"], data = nb)
                lt.remove(l)
        if l["name"] == "Biota_Smartarid":
            root_id = l["id"]

tree = tree.subtree(root_id)

for _ in range(5):
    for node in tree.leaves():
        if node.data is not None:
            if node.data == 0:
                tree.remove_node(node.identifier)

network_list = []
while len(tree) > 0:
    for leaf in tree.leaves():
        parent = tree.parent(leaf.identifier)
        if parent is not None:
            network_list.append([parent.tag, leaf.tag, leaf.data])
        tree.remove_node(leaf.identifier)

pd.DataFrame(network_list, columns=["parent", "child", "count"]).to_csv(export_file, index=False)






print("ok")


"""
report_paths = [r"D:\01_canyon_scale\data\habitat_coralfish\50_csv_image_label_report\333_csv_image_label_report\333-images-10m-p1.csv",
                r"D:\01_canyon_scale\data\habitat_coralfish\50_csv_image_label_report\335_csv_image_label_report\335-images-10m-p2.csv"]
report = pd.read_csv(report_paths[0])
report = pd.concat([report, pd.read_csv(report_paths[1])], axis=0)
report.reset_index(drop=True, inplace=True)



lt = api.get(f'label-trees/{lt_id}').json()["labels"]

tree = Tree()
tree.create_node("Root", "root")
while len(lt) > 0:
    for i, l in enumerate(lt):
        if l["name"] == "Habitats_CoralFish":
            cf_id = l["id"]
        print(f"{i}/{len(lt)}", end='\x1b[1K\r')
        if l["parent_id"] is None:
            tree.create_node(l["name"], l["id"], parent="root")
            lt.remove(l)
        else:
            if tree.contains(l["parent_id"]):
                nb = len(report[report['label_id'] == l["id"]])

                tree.create_node(f"{l['name']}: {nb}", l["id"], parent=l["parent_id"], data = nb)
                lt.remove(l)


tree = tree.subtree(cf_id)

for _ in range(5):
    for node in tree.leaves():
        if node.data is not None:
            if node.data == 0:
                tree.remove_node(node.identifier)
"""









