import pandas as pd
from ast import literal_eval
from datetime import datetime, timedelta
import os
from tqdm import tqdm
from decord import VideoReader
from decord import cpu
import geopandas as gpd
import numpy as np

dives = [
    {"name": "PL200-03",
     "video_dir": r"Z:\videos\CHEREEF_2021\CHE21_PL200-03",
     "report_file": r"D:\99_Tests\video\81-pl200-03.csv",
     "nav_file": r"I:\00_SIG\CHEREEF-2021\Plongees\Nav_Reference\ChEReef2021_HROV_pl200_NAV_pt.shp",
     },
    {"name": "PL201-04",
    "video_dir": r"Z:\videos\CHEREEF_2021\CHE21_PL201-04",
    "report_file": r"D:\99_Tests\video\88-pl201-04.csv",
    "nav_file": r"I:\00_SIG\CHEREEF-2021\Plongees\Nav_Reference\nav_reference201.shp",
     },
    {"name": "PL202-05",
    "video_dir": r"Z:\videos\CHEREEF_2021\CHE21_PL202-05",
     "report_file": r"D:\99_Tests\video\93-pl202-05.csv",
     "nav_file": r"I:\00_SIG\CHEREEF-2021\Plongees\Nav_Reference\nav_reference202.shp",
     },
    {"name": "PL203-06",
     "video_dir": r"Z:\videos\CHEREEF_2021\CHE21_PL203-06",
     "report_file": r"D:\99_Tests\video\94-pl203-06.csv",
     "nav_file": r"I:\00_SIG\CHEREEF-2021\Plongees\Nav_Reference\nav_reference203.shp",}


]

export_dir = r"D:\99_Tests\video"

for dive in dives:
    name = dive["name"]
    video_dir = dive["video_dir"]
    report_file = dive["report_file"]
    nav_file = dive["nav_file"]

    print("Processing dive ", name)

    nav = gpd.read_file(nav_file)
    nav = nav[['date','time', 'longitude', 'latitude', 'immersion']]
    if nav.dtypes['date'] == np.dtype('<M8[ms]'):
        nav["date"] = nav['date'].dt.strftime('%d/%m/%Y')
    nav['datetime'] = pd.to_datetime(nav['date'] + ' ' + nav['time'], format='%d/%m/%Y %H:%M:%S.%f')
    nav = nav.drop(columns=['date', 'time'])
    nav = nav.set_index('datetime')

    report = pd.read_csv(report_file)
    report  = report[report.shape_name != "WholeFrame"]

    report['frames'] = report.frames.apply(lambda x: literal_eval(str(x))[0])

    report = report[['label_name', 'label_hierarchy','video_filename', 'frames', "annotation_id"]]
    report = report.assign(latitude=0.0)
    report = report.assign(longitude=0.0)

    # Create video sample with frame labels
    for filepath in tqdm(os.listdir(video_dir)):
        if filepath.endswith(".mp4"):
            dt_str = filepath.split("_")[2]
            video_start_dt = datetime.strptime(dt_str, "%y%m%d%H%M%S")

            video_path = os.path.join(video_dir, filepath)

            vr  = VideoReader(video_path, ctx=cpu(0))
            fps_in = vr.get_avg_fps()

            annotations = report[report.video_filename == filepath]
            frames = list(set(annotations['frames'].to_list()))
            frames_dict = {x: int(round(x * fps_in)) for x in frames}

            for frame_s, frame_nb in frames_dict.items():

                frame_dt = video_start_dt + timedelta(seconds = frame_s)
                n = nav.iloc[nav.index.get_indexer([frame_dt], method='nearest')]

                if n.index[0] - frame_dt < timedelta(seconds=1):
                    report.loc[(report['frames'] == frame_s) & (report['video_filename'] == filepath), 'latitude'] = float(n.latitude.iloc[0])
                    report.loc[(report['frames'] == frame_s) & (report['video_filename'] == filepath), 'longitude'] = float(n.longitude.iloc[0])
                else:
                    print(f"Warning: no nav match for {name} frame {frame_s} at {frame_dt}")

    report.to_csv(os.path.join(export_dir, f"{name}_report.csv"), index=False)




