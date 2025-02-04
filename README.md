# Reprojection toolbox

Reprojection toolbox for Metashape

## Installation

Run the following command in Miniforge:

```bash
    mamba create -y -n reprojection_env -c conda-forge -c pytorch -c nvidia python=3.11 pyvista ultralytics pytorch torchvision torchaudio pytorch-cuda=11.8 tqdm pandas geopandas sqlalchemy scipy jupyterlab fiftyone sahi ipyfilechooser shapely pillow wandb treelib pyqt python-dotenv
    conda activate reprojection_env
```

Then install the latest version of Metashape python library from the [Agisoft website](https://www.agisoft.com/downloads/installer/).
(Or at ifremer, in the folder "Lep2\Imagerie\Metashape\Metashape-2.1.3-cp37.cp38.cp39.cp310.cp311-none-win_amd64.whl")

```bash
    pip install Metashape-2.1.3-cp37.cp38.cp39.cp310.cp311-none-win_amd64.whl
```
## Launch

```bash
    conda activate reprojection_env
    cd path/to/reprojection_toolbox
    python UI/app.py
```

## Usage

Config file: create a .rpj text file containing the following:

```txt
{
    'name': '',
    'project_directory': '',
    'video_directory': '',
    'image_directory': '',
    'metashape_project_path': '',
    'inference_model_path': '',
    'annotation_report_path': '',
    'reprojection_image_directory': '',
}
```

An example of a config file is provided in the example folder.


## Advanced development

Update the UI qt file:

```bash
    cd path/to/reprojection_toolbox/UI
    pyuic5 -x main.ui -o main_window.py
```
