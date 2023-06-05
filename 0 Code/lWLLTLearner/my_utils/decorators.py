import sys, time, logging
# Use these decorators by writing @<decorator_name> on top of the function.

# Function logger decorator.
def logged(function):
    """
    This decorator can be used to log the names of the decorated functions and 
    their return values.
    """
    def wrapper(*args, **kwargs):
        value = function(*args, **kwargs)

        with open('WLLTMetricLearner/logfiles/logfile.txt', 'a+') as f:
            f.write(f"{function.__name__}(...) =\t{value}\n")
        return value
    
    return wrapper

# Function timer decorator.
def timed(function):
    """
    This decorator can be used to log the execution time of 
    decorated functions.
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        value = function(*args, **kwargs)
        end = time.time()

        with open('WLLTMetricLearner/logfiles/timelogfile.txt', 'a+') as f:
            f.write(f"Time({function.__name__}) =\t{round((end-start)/1000, 5)} sec\n")

        return value

    return wrapper

def get_logger(filename='default', create_file=True, level=logging.DEBUG, formatter_verbose=None):
    logging.basicConfig(level=level, filename='dummy.log', filemode='w')
    # Create a logger for this module.
    logger = logging.getLogger(__name__)
    
    # Create a formatter for the creation of a customized logger.
    if formatter_verbose is None:
        formatter_verbose = logging.Formatter("%(asctime)s-%(levelname)s:\t\t%(message)s")
    formatter_short = logging.Formatter("LOG_%(levelname)s: %(message)s")

    # Create console handler for the logger.
    ch = logging.StreamHandler(sys.stdout)
    # Notice that the console handler will not print DEBUG-messages, 
    # since it is set to INFO in order to allow for nice overviews about the used kernels and datasets at runtime.
    ch.setLevel(level=logging.INFO)
    ch.setFormatter(formatter_short)
    if not logger.handlers:
        logger.addHandler(ch)

    return logger



@timed
@logged
def _test_function():
    result = 1

    for i in range(1, 30000):
        result *= i
    return result # Expected return value:

if __name__=='__main__':
    # print(_test_function())
    pass
