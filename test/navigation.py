import pandas as pd
import cv2
import os
from datetime import datetime, timedelta
from geopy.distance import geodesic
from tqdm import tqdm
import pandas as pd
from simpledbf import Dbf5

nav_file = r"Y:\CHEREEF_2025\ROV\ADELIE\pl884\nav_estime884.dbf"

video_dir = r"Y:\CHEREEF_2025\ROV\DATA\CHEREEF-2025-884-03\DATA\videos"
img_dest = r"D:\ChEReef25\reconstructions\falaise\complete\img\884-pl03"
nav_dest = r"D:\ChEReef25\reconstructions\falaise\complete\nav\nav_sampled_884_03_3.csv"

dt_start = datetime.strptime("2025-08-22 21:40:00", "%Y-%m-%d %H:%M:%S")
dt_end = datetime.strptime("2025-08-22 21:57:00", "%Y-%m-%d %H:%M:%S")

dbf = Dbf5(nav_file)
nav = dbf.to_dataframe()
nav['datetime'] = pd.to_datetime(nav['DATE'] + ' ' + nav['HEURE'], format='%d/%m/%Y %H:%M:%S')

mask = (nav['datetime'] > dt_start) & (nav['datetime'] <= dt_end)
nav = nav.loc[mask]

def compute_speed(index):
    w = 5
    if index < w//2:
        return None  # Not enough points yet

    segment = nav.iloc[index - w//2:index + w//2 ]

    if len(segment) == 0:
        return None

    total_distance = 0.0  # in meters
    for i in range(len(segment)-1):
        point1 = (segment.iloc[i]['LATITUDE'], segment.iloc[i]['LONGITUDE'])
        point2 = (segment.iloc[i + 1]['LATITUDE'], segment.iloc[i + 1]['LONGITUDE'])
        total_distance += geodesic(point1, point2).meters

    # Time delta in seconds
    time_delta = (segment.iloc[-1]['datetime'] - segment.iloc[0]['datetime']).total_seconds()
    if time_delta == 0:
        return 0.0

    return total_distance / time_delta  # m/s

nav.index = nav["datetime"]
nav.sort_index(inplace=True)
nav.reset_index(inplace=True, drop = True)
nav['speed'] = nav.index.to_series().apply(compute_speed)
nav = nav.dropna(subset=['speed'])
nav.index = nav["datetime"]

selection = [[0, nav.index[0]]]
while True:
    s = float(nav.iloc[selection[-1][0], nav.columns.get_loc("speed")])
    interval_s = int(0.9/(s+0.01) - 0.5)
    interval_s = max(min(interval_s, 5), 1) #min and max

    ind = nav.index.searchsorted(selection[-1][1] + timedelta(seconds=interval_s))
    if ind >= len(nav):
        break
    dt = nav.iloc[ind, nav.columns.get_loc("datetime")]
    selection.append([int(ind), dt])
id_selection = [selection[i][0] for i in range(len(selection))]
nav_sampled = nav.iloc[id_selection]

nav_sampled["label"] = nav_sampled.apply(lambda row: f"{row.datetime.strftime('%Y%m%dT%H%M%S.%f')}.jpg", axis=1)
nav_sampled["IMMERSION"] = -nav_sampled["IMMERSION"]

nav_sampled.to_csv(nav_dest)

# Video extraction
vid_list = []
for vid_filename in os.listdir(video_dir):
    if vid_filename.endswith("1.mp4"):
        dt_str = vid_filename.split("_")[2]
        dt = datetime.strptime(dt_str, "%y%m%d%H%M%S")

        if dt + timedelta(minutes=30) < dt_start or dt > dt_end:
            continue

        vid_list.append([vid_filename, dt, dt + timedelta(minutes=30)])

for video_file, vid_start, vid_stop in tqdm(vid_list):
    mask = (nav_sampled['datetime'] > vid_start) & (nav_sampled['datetime'] <= vid_stop)
    nav_video = nav_sampled.loc[mask]

    vidcap = cv2.VideoCapture(os.path.join(video_dir, video_file))
    success, image = vidcap.read()

    for dt in nav_video.index.tolist():
        ts = dt - vid_start
        img_file = os.path.join(img_dest, f"{dt.strftime('%Y%m%dT%H%M%S.%f')}.jpg")

        vidcap.set(cv2.CAP_PROP_POS_MSEC, (ts.seconds * 1000))
        success, image = vidcap.read()
        if success:
            cv2.imwrite(img_file, image)
        else:
            print(f"Failed to save {img_file}")





"""
img_list = []
for img_filename in os.listdir(image_dir):
    if img_filename.endswith(".jpg"):

        parts = img_filename.split('.')

        dt_str = parts[0]

        dt = datetime.strptime(dt_str, "%Y%m%dT%H%M%S")

        img_list.append([img_filename, dt])

img_pd = pd.DataFrame(img_list, columns=['filename', 'datetime'])

merged = pd.merge_asof(img_pd, nav, on="datetime", tolerance = timedelta(seconds=5), direction = "nearest")
merged = merged.dropna(subset=['LATITUDE', 'LONGITUDE'])

merged.index = merged['datetime']
merged.sort_index(inplace=True)
merged.reset_index(inplace=True, drop = True)


merged = merged.dropna(subset=['speed_5pts_mps'])
merged.index = merged['datetime']

selection = [[0, merged.index(0)]]
while selection[-1][0] < len(merged):
    s = merged.iloc(selection[-1][0])["speed_5pts_mps"]
    if s is not None:
        interval_s = int(0.9/s - 0.5)
    else:
        interval_s = 2.5
    ind = merged.index.searchsorted(selection[-1][1] + timedelta(seconds=interval_s))


exp_file = os.path.join(image_dir, "filt_nav.csv")
merged.to_csv(exp_file, index=False)

img_select_list = merged["filename"].to_list()

l = 0
for img_filename in tqdm(os.listdir(image_dir)):
    if img_filename.endswith(".jpg"):
        if img_filename not in img_select_list:
            #print("removing {}".format(img_filename))
            l+=1
            os.remove(os.path.join(image_dir, img_filename))

print(l/len(os.listdir(image_dir)))
"""
print("ok")
print("stop")