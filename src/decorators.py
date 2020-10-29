import time as t
from functools import wraps

def execution_time_calculator(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        start_time = t.time()
        result = function(*args, **kwargs)
        end_time = t.time() - start_time
        formatted_end_time = "{:.2f}".format((end_time / 60))
        print('FUNCTION: {} TIME ELAPSED : {}'.format(function.__name__, formatted_end_time))
        return result
    return wrapper

def start_time_calculator(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        start = t.localtime()
        start_time = t.strftime("%H:%M:%S", start)
        print('Start time is %s' % start_time)
        result = function(*args, **kwargs)
        return result
    return wrapper