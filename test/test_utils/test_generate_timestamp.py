import pytest
from source.utils.generate_timestamp import generate_timestamps

def test_generate_timestamps_invalid_window_size():
    " Test that an assertion is raised when the window size is invalid"
    start_training = "2023-01-01"
    i = 5
    window_size = 0  # This should trigger the assertion
    with pytest.raises(AssertionError, match="Window size must be greater than 0"):
        generate_timestamps(start_training, i, window_size)
