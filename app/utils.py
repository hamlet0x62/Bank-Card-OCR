from app.settings import UPLOADS_DIR, UPLOADS_URL_PREFIX
import os
from urllib.parse import urljoin

join_path = os.path.join


def url_for_uploads(filename):
    return urljoin(UPLOADS_URL_PREFIX, filename)


def get_uploads_filepath(filename):
    return join_path(UPLOADS_DIR, filename)


def get_file_ext(filename):
    splited = filename.rsplit('.', 1);
    if len(splited) > 1:
        return splited[0]
    else:
        return ''

