from pathlib import Path
from typing import Literal

from langchain_core.tools import ToolException
from crewai_tools import tool

DIRECTORY_NAME = 'village_wisdom_notebook'

def _no_caching_strategy(*args):
    return False


POSSIBLE_TASKS = ['initial plan', 'final solution']


@tool("get_wisdom_filename")
def get_wisdom_filename_tool(task: Literal[*POSSIBLE_TASKS]):
    """
    :param task: one of 'initial plan', 'final solution'
    :return: The filename in which the wisdom saves the plan
    """
    if task == 'initial plan':
        return _get_wisdom_initial_plan_filename()
    elif task == 'final solution':
        return _get_wisdom_final_plan_filename()
    else:
        raise ToolException(f"Illegal argument task: {task}. Must be one of {POSSIBLE_TASKS}")


def _get_wisdom_initial_plan_filename():
    return f"{DIRECTORY_NAME}/cross_river_initial_plan.txt"


def _get_wisdom_final_plan_filename():
    return f"{DIRECTORY_NAME}/cross_river_final_plan.txt"


POSSIBLE_FILE_NAMES = [_get_wisdom_initial_plan_filename(), _get_wisdom_final_plan_filename()]

@tool("write_wisdom_file")
def write_wisdom_file_tool(filename, contents):
    """
    Overwrites the contents to the specified filename. Any previous content will be replaced.
    :param filename: The target file name.
    :param contents: The content of the new file
    :return Confirmation of writing the file or error message
    """
    filepath = _get_file_path(filename)
    if str(filepath) not in POSSIBLE_FILE_NAMES:
        raise ToolException(f'Cannot overwrite {filepath}, only one of {POSSIBLE_FILE_NAMES}')

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(contents)

        return f"Successfully wrote contents to '{filepath}'."
    except IOError as e:
        raise ToolException(f"Error writing to file '{filepath}': {e}")


write_wisdom_file_tool.cache_function = _no_caching_strategy


@tool("read_wisdom_file")
def read_wisdom_file_tool(filename):
    """
    Returns the contents of the specified filename.
    :param filename: The file name to read.
    """
    if not filename:
        raise ToolException('First run get_wisdom_filename tool, and pass filename to read_wisdom_file tool')
    filepath = _get_file_path(filename)
    return filepath.read_text()


read_wisdom_file_tool.cache_function = _no_caching_strategy


@tool("delete_wisdom_file")
def delete_wisdom_file_tool(filename):
    """
    Deletes the specified file
    :param filename: The target file name to delete
    :return Confirmation of deleting the file or error message
    """
    filepath = _get_file_path(filename)
    assert str(filepath) in POSSIBLE_FILE_NAMES, f'cannot delete {filepath}, only one of {POSSIBLE_FILE_NAMES}'
    try:
        filepath.unlink()
        return f"The file {filepath} was deleted successfully."
    except FileNotFoundError:
        raise ToolException(f"Cannot delete the file {filepath}. It does not exist.")
    except IOError as e:
        raise ToolException(f"Error deleting file '{filepath}': {e}")


delete_wisdom_file_tool.cache_function = _no_caching_strategy


def _get_file_path(filename) -> Path:
    if not filename.endswith('.txt'):
        filename += '.txt'
    filename = Path(filename).name
    return Path(f'{DIRECTORY_NAME}/{filename}')

