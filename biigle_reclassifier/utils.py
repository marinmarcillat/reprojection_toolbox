import shutil
import requests
import os
from PIL import Image
import io
from Biigle.biigle import Api
from treelib import Tree


def list_images(dir):
    img_list = []
    for file in os.listdir(dir):  # for each image in the directory
        file_path = os.path.join(dir, file)
        if os.path.isfile(file_path):  # Check if is file
            filename, file_extension = os.path.splitext(file_path)
            if file_extension == '.jpg':
                img_list.append(file_path)
    return img_list

def connect(mail, token):
    try:
        return Api() if mail == "" or token == "" else Api(mail, token)
    except:
        return None

def save_label(api, annotation, old_label, new_label, cat):
    if cat == "Largo":
        ann = api.get(f'image-annotations/{annotation}/labels').json()
        if old_label != new_label and ann[0]['label']['id'] == old_label:
            p = api.post(
                f'image-annotations/{annotation}/labels',
                json={
                    'label_id': new_label,
                    'confidence': 1,
                },
            )
            p = api.delete(f"image-annotation-labels/{ann[0]['id']}")
    if cat == 'Label':
        label = api.get(f'images/{annotation}/labels').json()
        if old_label != new_label:
            for i in label:
                if i['label']['id'] == old_label:
                    p = api.post(f'images/{annotation}/labels', json={'label_id': new_label})
                    p = api.delete(f"image-labels/{ann[i]['id']}")
                    break


def download_largo(api, model, label, dir, video=False, label_name=""):
    media = 'video' if video else 'image'
    endpoint_url = '{}s/{}/{}-annotations/filter/label/{}'
    annotations = api.get(endpoint_url.format('volume', model, media, label)).json()

    if len(annotations) == 0:
        return 1

    if not video:
        patch_url = 'https://biigle.ifremer.fr/storage/largo-patches/{}/{}/{}/{}.jpg'
    else:
        patch_url = 'https://biigle.ifremer.fr/storage/largo-patches/{}/{}/{}/v-{}.jpg'

    for annotation_id, image_uuid in annotations.items():
        url = patch_url.format(image_uuid[:2], image_uuid[2:4], image_uuid, annotation_id)
        print('Fetching', url)
        patch = requests.get(url, stream=True)
        if not patch.ok:
            raise Exception(f'Failed to fetch {url}')
        if label_name != "":
            export_path = f'{dir}/{annotation_id}_{label_name}.jpg'
        else:
            export_path = f'{dir}/{annotation_id}.jpg'
        with open(export_path, 'wb') as f:
            patch.raw.decode_content = True
            shutil.copyfileobj(patch.raw, f)
    return 1

def download_image(api, volume, label, path):
    img_ids = api.get('volumes/{}/files/filter/labels/{}'.format(volume, label)).json()
    for i in img_ids:
        img = api.get('images/{}/file'.format(i))
        img_encoded = Image.open(io.BytesIO(img.content))
        out = img_encoded.resize((600, 400))
        out.save(os.path.join(path, str(i) + ".jpg"))
    return 1


def get_tree(lt):
    tree = Tree()
    tree.create_node("Root", "root")
    while len(lt) > 0:
        for l in lt:
            if l["parent_id"] is None:
                tree.create_node(l["name"], l["id"], parent="root")
                lt.remove(l)
            else:
                if tree.contains(l["parent_id"]):
                    tree.create_node(l['name'], l["id"], parent=l["parent_id"])
                    lt.remove(l)
                    if l["name"] == "Animalia":
                        animalia_id = l["id"]

    return tree.subtree(animalia_id)










