from flask.views import View, MethodView
from flask import send_from_directory, abort, make_response, request
import os.path as ospath
from functools import wraps
from werkzeug.exceptions import HTTPException
from werkzeug.wrappers import Response
import json


def no_cache(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        response = make_response(f(*args, **kwargs))
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response

    return decorated_function
