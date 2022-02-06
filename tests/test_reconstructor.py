from chaosformer.core.reconstructor import Reconstructor
from tests.logger import TestLogger
import pytest
import io


@pytest.mark.skip()
def print_to_string(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents


def test_reconstructor():
    logger = TestLogger()
    rec = Reconstructor()

    assert rec

    s = print_to_string(rec)
    logger.log(s)
    logger.write()