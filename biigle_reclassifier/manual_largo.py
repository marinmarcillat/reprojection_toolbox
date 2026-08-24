from Biigle.biigle import Api
import requests
import shutil
from tqdm import tqdm

api = Api()
output_dir = r"D:\99_Tests\autopolyp\Manual_largo\Extended"
label_id = 9813

patch_url = 'https://biigle.ifremer.fr/storage/largo-patches/{}/{}/{}/{}.jpg'

images = api.get("volumes/683/files").json()
for image_id in tqdm(images):
    image_uuid = api.get(f"images/{image_id}").json()['uuid']
    annotations = api.get(f"images/{image_id}/annotations").json()
    annotations = [annotation["id"] for annotation in annotations if annotation["labels"][0]["label"]["id"] == label_id]
    for annotation_id in annotations:
        url = patch_url.format(image_uuid[:2], image_uuid[2:4], image_uuid, annotation_id)
        #print('Fetching', url)
        patch = requests.get(url, stream=True)
        if not patch.ok:
            print(f'Failed to fetch {url}')
            continue

        export_path = f'{output_dir}/{annotation_id}.jpg'
        with open(export_path, 'wb') as f:
            patch.raw.decode_content = True
            shutil.copyfileobj(patch.raw, f)