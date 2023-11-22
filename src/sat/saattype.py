#: Import(s)
import dataclasses
from datetime import datetime, timedelta
from typing import List


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


@dataclasses.dataclass
class ProcessorParams:
    video_fp: str
    template_fp: str
    log_directory: str
    dbm_magnitude_threshold: float
    bgra_min_filter: List[float]
    bgra_max_filter: List[float]
    db_per_division: int
    total_num_db_divisions: int
    text_img_threshold: int
    frames_to_read_per_sec: int = 1
    center_freq_id: str = "CENTER"
    ref_lvl_id: str = "RL"
    span_lvl_id: str = "SPAN"
    scan_for_relative_threshold: bool = False,
    record_relative_signal: bool = True,
    record_scaled_signal: bool = True,
    record_max_signal_scaled: bool = False


@dataclasses.dataclass
class ProcessorQueueData:
    logs: List[str]
    current_frame: int
    finished: bool = False


if __name__ == "__main__":
    print(get_datetime_heading())
