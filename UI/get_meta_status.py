import Metashape

def get_chunk_id(doc, chunk_name):
    if chunk_name is not None:
        for i, chunk in enumerate(doc.chunks):
            if chunk.label == chunk_name:
                return i
    print("Warning ! Chunk not provided or not found, using first")
    return 0


def get_meta_status(qt, chunk_name):
    if qt.doc is None:
        doc = Metashape.Document()
        doc.open(qt.project_config["metashape_project_path"])
        qt.doc = doc
    else:
        doc = qt.doc

    if doc.read_only:
        print("Read only project")
        qt.project_config["read_only"] = True

    chunk_id = get_chunk_id(doc, chunk_name)

    chunk = doc.chunks[chunk_id]
    qt.metaChunk.setText(chunk.label)
    if chunk.tie_points is None:
        return "not_aligned"

    qt.tiePoints.setStyleSheet("QLabel {color : green; font-weight: bold}")
    qt.project_config["tie_points"] = True
    qt.project_config["aligned"] = True
    models = chunk.models
    for model in models:
        data_source = model.meta['BuildModel/source_data']
        if not data_source:
            data_source = 1
        else:
            data_source = int(data_source)
        if data_source == 0:
            qt.lowRes.setStyleSheet("QLabel {color : green; font-weight: bold}")
            qt.project_config["lowRes"] = True
        else:
            qt.highRes.setStyleSheet("QLabel {color : green; font-weight: bold}")
            qt.project_config["highRes"] = True
    return "aligned"





