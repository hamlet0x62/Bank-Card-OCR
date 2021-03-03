# coding:utf-8
from flask import request, jsonify
from functools import wraps, partial
import json


ERROR_CODE = -10


def error(msg, data=None):
    """
    send error json response
    :param msg:
    :return:
    """
    rv = {
        'code': ERROR_CODE,
        'msg': msg
    }
    if data:
        rv['data'] = data

    return rv


def _single_b64_item(request_data):
    b64 = request_data.get('b64', None)

    return b64 and not isinstance(b64, list)


def reject_invalid_request(condition_func, view_func):
    @wraps(view_func)
    def wrapper(*args, **kws):
        request_data = json.loads(request.data.decode("utf-8"))
        if condition_func(request_data):
            return view_func(request_data, *args, **kws)
        else:
            return jsonify(error('解析请求参数失败'))
    return wrapper


reject_not_single_b64 = partial(reject_invalid_request, _single_b64_item)
