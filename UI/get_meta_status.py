import Metashape


def get_meta_status(qt):
    chunk_id = int(qt.metaChunk.currentIndex())

    doc = Metashape.Document()
    doc.open(qt.project_config["metashape_project_path"])

    if doc.read_only:
        print("Read only project")
        qt.project_config["read_only"] = True
        return "read_only"

    for c in doc.chunks:
        qt.metaChunk.addItem(c.label)

    if len(doc.chunks) == 1:
        chunk = doc.chunk
    else:
        chunk = doc.chunks[chunk_id]

    if chunk.tie_points is not None:
        qt.tiePoints.setStyleSheet("QLabel {color : green; font-weight: bold}")
        qt.project_config["tie_points"] = True
    else:
        return "not_aligned"

    qt.project_config["aligned"] = True
    models = chunk.models
    for model in models:
        if int(model.meta['BuildModel/source_data']) == 0:
            qt.lowRes.setStyleSheet("QLabel {color : green; font-weight: bold}")
            qt.project_config["lowRes"] = True
        else:
            qt.highRes.setStyleSheet("QLabel {color : green; font-weight: bold}")
            qt.project_config["highRes"] = True
    return "aligned"





