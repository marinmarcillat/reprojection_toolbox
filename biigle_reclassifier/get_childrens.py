from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
import sys
from tqdm import tqdm


from Biigle.biigle import Api
from biigle_reclassifier.choose_label import SelectWindow
from biigle_reclassifier.utils import download_largo, get_tree

project_id = 26
label_tree_id = 60

api = Api()

if api is None:
    print("API connection failed")
    exit(1)

label_tree = api.get(
    f'label-trees/{label_tree_id}'
).json()["labels"]

app = QApplication(sys.argv)

dir_path = QFileDialog.getExistingDirectory(None, 'Saving directory', r"")
if dir_path == "":
    exit(1)

sel = SelectWindow(label_tree)
res = sel.exec_()
if res == QDialog.Accepted:
    label_id, label_name = sel.get_value()
else:
    exit(1)

lt = label_tree.copy()
tree= get_tree(lt)
sub_t = tree.subtree(int(label_id))

label_dict = {label_id: label_name}
for node in sub_t.all_nodes():
    label_dict[node.identifier] = node.tag

vol_ids = [vol["id"] for vol in api.get(f'projects/{project_id}/volumes').json()]

for vol_id in tqdm(vol_ids):
    for l_id, l_name in label_dict.items():
        download_largo(api, vol_id, l_id, dir_path, video = True, label_name=l_name)
    print(f"Downloaded volume {vol_id}")

print("Download complete")