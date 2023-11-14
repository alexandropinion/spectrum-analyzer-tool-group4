#!/usr/bin/env python3

#: Imports
import logging
from gui import start


#: Main entry point
def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s;%(levelname)s;%(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    start()


if __name__ == "__main__":
    main()
