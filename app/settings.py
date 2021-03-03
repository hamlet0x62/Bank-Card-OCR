import os

_join_path = os.path.join

PROJ_DIR = os.path.dirname(__file__)
STATIC_FILE_DIR = _join_path(PROJ_DIR, 'static')
UPLOADS_DIR = _join_path(PROJ_DIR, 'static', 'uploads')

UPLOADS_URL_PREFIX = '/uploads/'

