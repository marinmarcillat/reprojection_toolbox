from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
import sys
from tqdm import tqdm


from Biigle.biigle import Api
from biigle_reclassifier.choose_label import SelectWindow
from biigle_reclassifier.utils import download_largo, get_tree
import os

project_id = 50
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
    label_dict[node.identifier] = ''.join(e for e in node.tag if e.isalnum())

vol_ids = [vol["id"] for vol in api.get(f'projects/{project_id}/volumes').json()]
#vol_ids = [462]

for vol_id in tqdm(vol_ids):
    for l_id, l_name in label_dict.items():
        store_dir = os.path.join(dir_path, l_name)
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)
        download_largo(api, vol_id, l_id, store_dir, video = False, label_name=l_name)
        if not os.listdir(store_dir):
            os.rmdir(store_dir)
    print(f"Downloaded volume {vol_id}")

print("Download complete")