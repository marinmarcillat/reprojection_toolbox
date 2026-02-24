# Reprojection toolbox

Reprojection toolbox for Metashape.
This is an experimental software, all the features are not fully implemented yet and may not work as expected.
Please report any bugs or issues you encounter while using the toolbox.

## Tools description

Mandatory parameters:
- name: name of the project
- project_directory: path to the project directory where all the outputs will be saved

### Reconstruction
The reconstruction tool allows you to reconstruct a 3D model from a set of images in Metashape.

Requires the following parameters:
- image_directory: path to the directory containing the images to be used for reconstruction

### Non overlapping image selection
The non overlapping image selection tool allows you to select a subset of images that do not overlap with each other. 
This can be useful for reducing the number of images to be annotated and the number of duplicated annotations

Requires reprojection beforhand, or the following parameters:
- metashape_project_path: path to the Metashape project file (.psx)

### Inference
The inference tool allows you to run inference on the selected images using a pre-trained model.
Requires the following parameters:
- inference_model_path: path to the pre-trained model to be used for inference

### Reprojection
The reprojection tool allows you to reproject the annotations from the selected images onto the 3D model, and filter the duplicates.
Requires the following parameters:
- metashape_project_path: path to the Metashape project file (.psx)
- annotation_path

## Installation

Download the latest release from the releases page.
Run the RT_setup.exe file to install the toolbox. 
This will install the necessary dependencies and set up the environment for the toolbox to run.

## Usage

Create a new .rpj config file using the file -> new and fill in the required fields.
