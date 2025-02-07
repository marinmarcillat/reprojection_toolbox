import Metashape


def get_meta_status(qt):
    chunk_id = int(qt.metaChunk.currentIndex())

    doc = Metashape.Document()
    doc.open(qt.project_config["metashape_project_path"])

    if doc.read_only:
        print("Read only project")
        qt.project_config["read_only"] = True

    for c in doc.chunks:
        qt.metaChunk.addItem(c.label)

    chunk = doc.chunk if len(doc.chunks) == 1 else doc.chunks[chunk_id]
    if chunk.tie_points is None:
        return "not_aligned"

    qt.tiePoints.setStyleSheet("QLabel {color : green; font-weight: bold}")
    qt.project_config["tie_points"] = True
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





