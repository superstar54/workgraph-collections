import pytest
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    """Run pytest and exit with the correct status code."""
    pytest_args = ["tests/ase/test_ase_espresso.py"]

    logging.info(f"Running tests with arguments: {pytest_args}")

    # Run pytest with extracted arguments
    result = pytest.main(pytest_args)

    if result.value != 0:
        logging.error(f"Tests failed with exit code: {result.value}")
        sys.exit(result.value)
    else:
        logging.info("All tests passed successfully.")


main()
