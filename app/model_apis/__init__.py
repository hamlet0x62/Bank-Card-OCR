# coding: utf-8
import requests
import json
import numpy as np
import time
from functools import wraps

from app.model_apis.utils.rpn_msr.proposal_layer import proposal_layer
from app.model_apis.utils.text_connector.detectors import TextDetector

_base_model_url = "http://localhost:8501/v1/models"

_detect_url = "{base_url}/detection:predict".format(base_url=_base_model_url)
_recognize_url = "{base_url}/recognize:predict".format(base_url=_base_model_url)


def add_elapsed_time(func):
    @wraps(func)
    def wrapper(*args, **kws):
        start = time.time()
        rst = func(*args, **kws)
        if isinstance(rst, dict):
            rst['elapsed'] = "{:.2f}".format(time.time()-start)
        return rst
    return wrapper


def add_ok_status(func):
    """
    为函数的执行结果增加 'ok' key，
    给被装饰函数注入 'ok_func'函数，
    若被装饰函数希望反馈执行成功信息，
    只需在其内部执行 'ok_func'，即可
    在其返回结果中使 'ok' 的 value
    为1， 若不执行'ok_func'则'ok'
    的value始终为0，表示执行不成功。
    :param func:
    :return:
    """
    @wraps(func)
    def wrapper(*args, **kws):
        outputs = {'ok': 0}

        def ok_func():
            outputs['ok'] = 1
        rst = func(*args, ok_func, **kws)
        outputs.update(rst)
        return outputs
    return wrapper


def catch_error(err_cls, msg=None):
    def outer(func):
        @wraps(func)
        def inner(*args, **kws):
            try:
                rst = func(*args, **kws)
                return rst
            except err_cls as err:
                rv = {"err": err, 'ok': 0}
                if msg:
                    rv['msg'] = msg
                return rv
        return inner
    return outer


@add_elapsed_time
@catch_error(requests.exceptions.ConnectionError, msg="卡号区域检测接口调用失败")
def detect(data):
    rst = requests.post(_detect_url, data=data)
    return parse_detection_result(rst.content) if rst.status_code == 200 else parse_error_resp(rst)


@add_elapsed_time
@catch_error(requests.exceptions.ConnectionError, msg="卡号识别接口调用失败")
def recognize(data):
    rst = requests.post(_recognize_url, data=data)
    return parse_recognize_result(rst.content) if rst.status_code == 200 else parse_error_resp(rst)


def build_b64_item(encoded):
    return {"b64": encoded}


def build_singe_b64_req_data(encoded):
    return json.dumps({"inputs": build_b64_item(encoded)})


def build_b64_array_req_data(encoded_list):
    b64_list = [build_b64_item(encoded) for encoded in encoded_list]
    req_data = {"inputs": b64_list}

    return json.dumps(req_data)


@add_ok_status
def parse_detection_result(detect_rst, ok):
    """
    解析tensorflow/serving所返回的
    卡号区域检测结果,只解析返回结果中
    得分最高的一个区域
    :param bytes detect_rst: 卡号区域检测结果
    :return: dict
    """
    detect_result = json.loads(detect_rst.decode('utf-8'))
    detect_output = detect_result['outputs']

    cls_prob, bbox_pred = [np.array(detect_output[k], dtype=np.float32) for k in ['cls_prob', 'bbox_pred']]
    im_shape = detect_output['im_shape'][1:]
    textsegs, _ = proposal_layer(cls_prob, bbox_pred, np.array(im_shape).reshape([1, 3]))
    scores = textsegs[:, 0]
    textsegs = textsegs[:, 1:5]

    textdetector = TextDetector(DETECT_MODE='H')
    boxes = textdetector.detect(textsegs, scores[:, np.newaxis], im_shape[:2])
    boxes = np.array(boxes, dtype=np.int)
    if len(boxes) == 0:
        return {}
    ok()
    xmin, ymin, _, _, xmax, ymax, *_ = boxes[0]
    h, w = im_shape[:2]

    return {'xmin': xmin/w, 'ymin': ymin/h,
            'xmax': xmax/w, 'ymax': ymax/h,
            'score': float(scores[0]), 'im_info': im_shape}


@add_ok_status
def parse_recognize_result(recognize_result, ok):
    recognize_rst = json.loads(recognize_result.decode('utf-8'))
    recognize_outputs = recognize_rst['outputs']
    output = dict()
    if len(recognize_outputs) == 0 or len(recognize_outputs[0]) == 0:
        return output
    output['result'] = recognize_outputs[0]
    ok()

    return output


def parse_error_resp(resp):
    return {
        'remote_status_code': resp.status_code,
        'content': resp.content.decode('utf-8')
    }


if __name__ == "__main__":
    from base64 import b64encode
    test_img_path = r'/Users/simon/Desktop/Bank_Card_Ocr/detection/data/demo/200808061016138636.jpg'
    with open(test_img_path, 'rb') as f:
        img_bytes = f.read()

    b64encoded = b64encode(img_bytes).decode('utf-8')
    req_data = build_singe_b64_req_data(b64encoded)

    print(detect(req_data))
