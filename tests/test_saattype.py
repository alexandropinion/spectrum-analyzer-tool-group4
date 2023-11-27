from src.sat import saattype


def test_get_datetime_heading():
    assert len(saattype.get_datetime_heading()) >= 8


def test_convert_seconds_to_datetime_hour_min_sec_ms():
    timestr: str = saattype.convert_seconds_to_datetime_hour_min_sec_ms(10000000.0)
    assert timestr == "17:46:40:000"
