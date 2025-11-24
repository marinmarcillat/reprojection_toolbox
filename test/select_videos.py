import os
from datetime import datetime, timedelta



video_dir = r"Y:\REDECOR_2025\DATA_ROV\BRUTES\REDECOR-2025-878-02\DATA\videos"

result = []
for file in os.listdir(video_dir):
    if file.endswith("1.mp4"):
        print(file)
        dt_start = datetime.strptime(file.split("_")[-2], "%y%m%d%H%M%S")
        dt_end = dt_start + timedelta(minutes=30)
        result.append([file, dt_start, dt_end])


print("ok")
print("stop")

