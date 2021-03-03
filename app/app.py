# coding: utf-8
from flask import Flask, jsonify, render_template
from flask_cors import cross_origin

from app.settings import UPLOADS_DIR, UPLOADS_URL_PREFIX
from app.patch import SharedDataMiddlewareWithCors as SharedDataMiddleware
import app.model_apis as apis
import app.response_utils as response_utils
import app.ext as ext


def create_app(config=None):
    app = Flask(__name__, template_folder="./template", static_folder="./static")
    app.config.from_pyfile("config.py")

    @app.route('/')
    def get_index():
        return render_template("index.html")

    @app.route('/detect', methods=['POST'])
    @cross_origin()
    @response_utils.reject_not_single_b64
    def detect_bank_card(request_data):
        b64 = request_data['b64']
        detect_req_data = apis.build_singe_b64_req_data(b64)
        rst = apis.detect(detect_req_data)
        if rst.get('ok', None):
            return jsonify(rst)
        else:
            return jsonify(response_utils.error('检测失败.', data=rst))

    @app.route('/recognize', methods=['POST'])
    @cross_origin()
    @response_utils.reject_not_single_b64
    def recognize_bank_cardno(request_data):
        b64 = request_data['b64']
        recognize_req_data = apis.build_singe_b64_req_data(b64)
        rst = apis.recognize(recognize_req_data)

        if rst.get('ok', None):
            return jsonify(rst)
        else:
            return jsonify(response_utils.error('识别失败', data=rst))

    ext.init_app(app)
    app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
        UPLOADS_URL_PREFIX: UPLOADS_DIR
    })

    return app


app = create_app()

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000)
