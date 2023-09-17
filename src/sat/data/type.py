#: Import(s)
from datetime import datetime


#: Function(s)
def get_datetime_heading() -> str:
    now = datetime.now()
    dt_string = now.strftime("(%d_%m_%Y)-[%H-%M-%S]")
    return dt_string
    #: Main entry point


if __name__ == "__main__":
    print(get_datetime_heading())
