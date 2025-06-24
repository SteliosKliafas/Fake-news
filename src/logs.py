import logging
from dotenv import load_dotenv
import os


load_dotenv()
LOG_LEVEL = 0

# # https://www.delftstack.com/howto/python/python-logging-to-file-and-console/
# # Use the logging Module to Print the Log Message to File and Console in Python
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler("debug.log"),
#         logging.StreamHandler(sys.stdout)
#     ]
# )

# TODO also change colors
# DEFAULT VALUES
NOTSET = 0
DEBUG = 10  # for debug purposes
INFO = 20  # instead of print
WARN = 30
ERROR = 40  # user/ input errors
CRITICAL = 50  # system errors

# EXTEND LOG LEVEL
DEV_INFO = 18  # use to print extra information
SUCCESS = 25
logging.addLevelName(DEV_INFO, 'DEV_INFO')  # extend log level
logging.addLevelName(SUCCESS, 'SUCCESS')  # extend log level


def dev_info(self, message, *args, **kws):  # extend log level
    self.log(DEV_INFO, message, *args, **kws)  # extend log level


def success(self, message, *args, **kws):  # extend log level
    self.log(SUCCESS, message, *args, **kws)  # extend log level


logging.Logger.dev_info = dev_info  # extend log level
logging.Logger.success = success  # extend log level


# Use the logging Module to Print the Log Message to Console in Python
class ColorFormatter(logging.Formatter):
    grey = "\x1b[1m"
    green = "\x1b[1;32m"
    blue = '\x1b[38;5;39m'
    cyan = '\x1b[1;36m'
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    basic_format = "%(asctime)s [%(levelname)s]  %(message)s"

    FORMATS = {
        logging.DEBUG: basic_format + reset,
        logging.INFO: grey + basic_format + reset,
        logging.WARNING: yellow + basic_format + reset,
        logging.ERROR: red + basic_format + reset,
        logging.CRITICAL: bold_red + basic_format + reset,
        SUCCESS: green + basic_format + reset,
        DEV_INFO: cyan + basic_format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


ch = logging.StreamHandler()
ch.setLevel(DEBUG)
ch.setFormatter(ColorFormatter())

# logging.basicConfig(format="[%(levelname)s] %(message)s")
extended_logger = logging.getLogger()  # extend log level
extended_logger.setLevel(DEBUG)
for h in extended_logger.handlers:
    extended_logger.removeHandler(h)

extended_logger.addHandler(ch)


# extended_logger.addHandler(ch)
def color_test():
    extended_logger.setLevel(DEBUG)
    extended_logger.debug("COLOR TEST")
    extended_logger.debug("debug")
    extended_logger.dev_info('dev_info')
    extended_logger.info("info")
    extended_logger.success('success')
    extended_logger.warning("warning")
    extended_logger.error("error")
    extended_logger.critical("critical")
    print("this is the print")


# change log level according to dev or production env
if LOG_LEVEL == 'PRODUCTION':
    extended_logger.setLevel(INFO)
else:
    extended_logger.setLevel(DEV_INFO)


if __name__ == '__main__':
    color_test()
