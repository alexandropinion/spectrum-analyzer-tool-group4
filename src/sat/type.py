#: Import(s)
from datetime import datetime, timedelta


#: Function(s)
def get_datetime_heading() -> str:
    now = datetime.now()
    dt_string = now.strftime("(%d_%m_%Y)-[%H-%M-%S]")
    return dt_string
    #: Main entry point


def convert_seconds_to_datetime_hour_min_sec_ms(seconds: float) -> str:
    delta = timedelta(seconds=seconds)
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = delta.microseconds // 1000
    time_format = "{:02}:{:02}:{:02}:{:03}"
    return time_format.format(hours, minutes, seconds, milliseconds)


if __name__ == "__main__":
    print(get_datetime_heading())
