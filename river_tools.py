import random
from typing import Set, List

from crewai_tools import tool
from langchain_core.tools import ToolException

left_side: Set[str] = set()
right_side: Set[str] = set()
execution_trace: List[str] = []
max_shout_len: int = 20


def _no_caching_strategy(*args):
    return False


def _concat_str(strings):
    strings = [f"a {s}" for s in strings]
    if len(strings) == 0:
        return ""
    elif len(strings) == 1:
        return strings[0]
    elif len(strings) == 2:
        return " and ".join(strings)
    else:
        return ", ".join(strings[:-1]) + ", and " + strings[-1]


def _get_farmer_other_sides():
    farmer_side = right_side if 'farmer' in right_side else left_side
    assert 'farmer' in farmer_side
    other_side = left_side if 'farmer' not in left_side else right_side
    assert 'farmer' not in other_side
    return farmer_side, other_side


@tool('cross_river')
def cross_river_tool(item: str = None) -> str:
    """This boat can be used by a farmer to take themselves and at most one item across the river to the bank on the other side."""
    farmer_side, other_side = _get_farmer_other_sides()

    if item in {"farmer", "none", "", "alone"}:
        item = None
    if item is not None and item not in farmer_side:
        raise ToolException(f'{item} is not found on the same river bank as the farmer.')

    update = "The farmer crosses the river"
    other_side.update({"farmer"})
    farmer_side.difference_update({"farmer"})

    if not item:
        update += " alone"
    else:
        update += f" with a {item}"
        other_side.update({item})
        farmer_side.difference_update({item})
    update += f" to {'right' if 'farmer' in right_side else 'left'} bank."

    if "goat" in farmer_side:
        assert "farmer" not in farmer_side
        eat_cabbage = "cabbage" in farmer_side
        eat_goat = "wolf" in farmer_side
        if eat_cabbage:
            update += "The goat eats the cabbage since they are alone without the farmer."
            farmer_side.difference_update({"cabbage"})
        if eat_goat:
            update += "The wolf eats the goat since they are alone without the farmer."
            farmer_side.difference_update({"goat"})

    execution_trace.append(update)
    # without the riverbank states since it could be irrelevant when used in a different run.

    #update += f" {_describe_river_sides()}"
    return update


cross_river_tool.cache_function = _no_caching_strategy


@tool('scout')
def scout_tool(*args, **kwargs):
    """
     This tool returns which item is on which river bank.
     It takes no input parameters whatsoever.
     It does not move the farmer nor any other item.
    """
    assert not args and not kwargs, "This tool takes no input parameters."
    lb = f"left bank has {_concat_str(left_side)}" if left_side else "left bank is empty"
    rb = f"right bank has {_concat_str(right_side)}" if right_side else "right bank is empty"
    return f"Now the {lb} while the {rb}."


scout_tool.cache_function = _no_caching_strategy


@tool('exeuction tracing')
def execution_tracing_tool():
    """Recounts all the steps the farmer has executed"""
    return execution_trace


@tool("shout")
def shout_tool(word: str):
    """
    A tool used to frighten off a wolf.
    Works only if the farmer and the wolf are at the same side of the river.
    :param word: The capitalized onomatopoeic word to shout out loud.
    """
    if word.upper() != word:
        raise ToolException("The shout word must be all upper case.")

    farmer_side, other_side = _get_farmer_other_sides()

    if 'wolf' in other_side and 'wolf' not in farmer_side:
        raise ToolException("The wolf is too far away at the other side. Shouting wouldn't work at that distance.")

    if 'wolf' not in farmer_side:
        assert 'wolf' not in other_side
        raise ToolException("There is no wolf here to shout at.")

    probability = 1.0 * min(max_shout_len, len(word)) / max_shout_len

    update = f"The farmer shouts {word} "
    if random.random() > probability:
        update += f'but fails to frighten the wolf. They should try shouting a longer word up to {max_shout_len} characters!'
    else:
        farmer_side.difference_update({'wolf'})
        update += f'and successfully frightens away the wolf.'

    return update


shout_tool.cache_function = _no_caching_strategy

