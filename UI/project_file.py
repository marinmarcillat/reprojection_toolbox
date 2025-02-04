import json

project_template = {
    'name': '',
    'project_directory': '',
    'video_directory': '',
    'image_directory': "",
    "metashape_project_path": "",

    'inference_model_path': '',
    'annotation_report_path': '',
    "reprojection_image_directory": "",
}


def read_json(file_path):
    """
    Reads a JSON file and returns its contents as a dictionary.

    Args:
        file_path: The path to the JSON file.

    Returns:
        dict: The contents of the JSON file as a dictionary, or None if an error occurs.
    """
    #try:
    with open(file_path, "r") as f:
        return json.load(f)
    #except Exception:
    #    print("Error reading, please check your project json file and try again.")
    #    return None


def write_json(file_path, project_config):
    """
    Writes a dictionary to a JSON file.

    Args:
        file_path: The path to the JSON file.
        project_config: The dictionary to be written.

    Returns:
        None.
    """
    with open(file_path, "w") as outfile:
        json.dump(project_config, outfile, indent=4)


def create_json(name, directory):
    """
    Creates a new JSON file with the specified name and directory.

    Args:
        name: The name of the project.
        directory: The directory where the JSON file will be created.

    Returns:
        dict: The project configuration as a dictionary.
    """
    project_path =  f'{name}.json'
    project_config = project_template.copy()
    project_config['name'] = name
    project_config['project_directory'] = directory

    write_json(project_path, project_config)
    return project_config