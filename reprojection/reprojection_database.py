from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, ForeignKey, Float
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

import os

Base = declarative_base()

class Camera(Base):
    __tablename__ = 'Cameras'

    name = Column(String(50), primary_key=True)
    abs_path = Column(String(100), nullable=False)

    poly_hull_file = Column(String(100))
    center_x = Column(Float)
    center_y = Column(Float)
    center_z = Column(Float)
    annotations = relationship("Annotation", backref="camera")



class Annotation(Base):
    __tablename__ = 'Annotations'

    id = Column(Integer, primary_key=True)
    type = Column(String(20))
    confidence = Column(Float)
    label = Column(String(50))

    polygon_3D_file = Column(String(60))
    tie_point_file = Column(String(60))
    misses_percent = Column(Float)

    camera_name = Column(String(50), ForeignKey('Cameras.name'))
    individual_id = Column(Integer, ForeignKey("Individuals.id"), nullable=True)


class Individual(Base):
    __tablename__ = 'Individuals'
    id = Column(Integer, primary_key=True)
    annotations = relationship("Annotation", backref="individual")


def open_reprojection_database_session(db_dir, create: bool, db_name='reprojection'):

    db_path = os.path.join(db_dir, f'{db_name}.db')

    if not os.path.exists(db_dir) and not create:
        raise FileNotFoundError(f"Database directory {db_dir} does not exist")

    if create and os.path.exists(db_path):
        os.remove(db_path)
    engine = create_engine(f'sqlite:///{db_path}', echo=False)
    if create:
        Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=True)(), db_path


def get_db_status(qt, session):
    if len(session.query(Individual).all()) != 0:
        qt.project_config["individual"] = True
    if len(session.query(Annotation).all()) != 0:
        qt.project_config["reprojected"] = True



if __name__ == '__main__':
    db_dir = r"D:\tests\databases"
    session, path = open_reprojection_database_session(db_dir, True)

    print("session")
