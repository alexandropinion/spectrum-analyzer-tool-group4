#!/usr/bin/env python3

#: Imports
import logging

logging.basicConfig(format='%(asctime)s - %(funcName)s - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    datefmt="%Y-%m-%d %H:%M:%S")
from gui import start


#: Main entry point
def main() -> None:
    start()


if __name__ == "__main__":
    main()
