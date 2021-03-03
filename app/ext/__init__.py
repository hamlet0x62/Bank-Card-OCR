from flask_cors import CORS

allow_all = {'origins': '*'}

resources = {r'/static/*': allow_all}
cors = CORS(resources=resources)


def init_app(app):
    cors.init_app(app)

