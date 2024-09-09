import logging
import sys

LOGGER = logging.getLogger("logger")
LOGGER.setLevel(logging.INFO)
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(logging.Formatter(fmt="[%(asctime)s: %(levelname)s] %(message)s"))
LOGGER.addHandler(handler)
