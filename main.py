from textwrap import dedent
from typing import List, Literal

from crewai import Task, Agent
from pydantic import RootModel

from llms import llm
from river_tools import cross_river_tool, execution_tracing_tool, scout_tool, left_side, right_side, shout_tool
from wisdom_tools import write_wisdom_file_tool, get_wisdom_filename_tool, delete_wisdom_file_tool, \
    read_wisdom_file_tool

WOLVES = True


def plan_execute_reflect():
    left_side.update(["farmer", "cabbage", "goat"])

    if WOLVES:
        left_side.update(["wolf"])
        right_side.update(["wolf"])

    verbose = True
    model = llm()

    village_wisdom = Agent(
        role="Village Wisdom",
        goal="Creating plans that solves problems step by step.",
        backstory=dedent("""
        The Village Wisdom is strong-willed. They like solving problems, by breaking them into steps.
        Their prompt instruction always starts with "Let's think step by step". 
        """),
        llm=model,
        verbose=verbose,
        memory=False,
    )

    read_final_plan_task = Task(
        description="""
            Read the final plan file, in order to check if it is faulty. You MUST not attempt to solve the puzzle in the file.
            If the file does not include the problem details it is faulty and must be deleted.
            If the file includes an empty final plan it is faulty and must be deleted.
            Otherwise return the final plan file contents.
        """,
        expected_output='The complete contents of the final plan file, prefixed by "This is the output of the previous run:\n" if the file exists',
        agent=village_wisdom,
        tools=[
            get_wisdom_filename_tool,
            read_wisdom_file_tool,
            delete_wisdom_file_tool,
        ]
    )
    read_final_plan_task.execute()

    river_planning_task = Task(
        description="""
        Your task is to create and persist an initial plan. 
        Please follow the instructions specified one step at a time, completing each step before moving on to the next:
        - Scout the river.
        - If the final plan of the previous run was provided in the context then trust that it works. The final plan is now our new initial plan. Do not try adding any more steps to it.         
        - If the final plan was not provided then solve the specified problem and produce an initial plan that includes complete step by step instructions formatted as a hyphenated sequence. 
        - Once the initial plan is ready, fill the following template and save the result to the initial plan file: "Problem Details:\n{problem details}\nInitial Plan:\n{initial plan}"
        Problem Details:
        A farmer wants to cross a river and take with him a goat, and a cabbage. 
        There is a boat that can fit himself plus one item.
        If the goat and the cabbage are alone on the shore, the goat will eat the cabbage. 
        How can the farmer bring themselves, the boat, and all items across the river?
        """,
        expected_output="A plan as ordered actions the farmer needs to take. Each of the plan's steps contain a single crossing of the river, not a roundtrip.",
        context=[read_final_plan_task],
        agent=village_wisdom,
        tools=[
            scout_tool,
            get_wisdom_filename_tool,
            write_wisdom_file_tool,
        ]
    )

    river_planning_task.execute()

    the_farmer = Agent(
        role="The Farmer",
        goal="""Getting to the right riverbank with the goat, and the cabbage.""",
        backstory=dedent("""
            The Farmer is isolated and reclusive.
            They are skilled in farming and raising livestock, such as cabbage and goats.                                    
            """),
        llm=model,
        verbose=verbose,
        memory=False,
    )

    the_farmer_tools = [
        get_wisdom_filename_tool,
        read_wisdom_file_tool,
        cross_river_tool,
        scout_tool,
        shout_tool
    ]

    river_crossing_task_desc = """
        Please follow the instructions specified one step at a time, completing each step before moving on to the next.
        If a step outputs a capitalized word, do not proceed with any more steps:
    """
    if not WOLVES:
        river_crossing_task_desc += """
            1. perform the step: "{step}"
            2. Use the scout tool
            3. If the farmer, goat, and cabbage are on the right riverbank then output SUCCESS. Otherwise output CONTINUE.
        """
    else:
        river_crossing_task_desc += """
            1. Use the scout tool.
            2. Shout as long as there is a wolf at the same riverbank as the farmer and the goat.
            3. If you failed to scare the wolf aware then output FAILURE.
            4. If there is no wolf at the same riverbank as the farmer, perform the step: "{step}"
            5. After performing the step, if the farmer, goat, and cabbage are on the right riverbank then output SUCCESS, otherwise output CONTINUE.
        """

    river_crossing_steps = Task(
        description=dedent("""
            Read the initial plan from the wisdom notebook. 
            Parse the plan into a JSON list of steps without invoking any additional tool.
            Each step must be stripped of numbering prefixes, hyphens, or single quotes, and must be surrounded by double quotes.
        """),
        expected_output='A list of steps',
        output_pydantic=RootModel[List[str]],
        agent=the_farmer,
        tools=[get_wisdom_filename_tool, read_wisdom_file_tool],
    ).execute().root

    for step in river_crossing_steps:
        step_task_description = river_crossing_task_desc.format(step=step)
        print(f"Creating task for farmer:\n {step_task_description}")
        step_output = Task(
            description=step_task_description,
            expected_output='A quoted value of either "CONTINUE" or "SUCCESS" or "FAILURE"',
            output_pydantic=RootModel[Literal['CONTINUE', 'SUCCESS','FAILURE']],
            agent=the_farmer,
            tools=the_farmer_tools,
        ).execute().root

        if step_output in ['SUCCESS', 'FAILURE']:
            break
        assert step_output == 'CONTINUE', f"unknown step_result {step_output}"

    river_crossing_validation_result = Task(
        description=dedent("""
               First Scout.
               Then, if the farmer, the cabbage, and the goat are on the right river bank output SUCCEEDED, otherwise output FAILED        
               """),
        expected_output='A quoted value of either "SUCCEEDED" or "FAILED"',
        output_pydantic=RootModel[Literal['SUCCEEDED', 'FAILED']],
        agent=village_wisdom,
        tools=[scout_tool]
    ).execute().root

    if river_crossing_validation_result == 'FAILED':
        river_crossing_reflection_task_description = dedent("""
            Delete the final plan file.
        """)
    else:
        river_crossing_reflection_task_description = """
            Use the execution tracing tool to list all the farmer's execution steps. Format the list as a hyphenated sequence. This list is now the new plan. 
            If a final plan does not exist or if the final plan is longer than the new plan:
                Extract the content of the "Problem Details:" section of the initial plan,
                and fill the following template and save to the final plan notebook: "Problem Details:\n{problem details}\nFinal Plan:\n{new plan}"         
        """

    river_crossing_reflection_task = Task(
        description=river_crossing_reflection_task_description,
        expected_output="The contents of the notebook final plan file if exists.",
        agent=village_wisdom,
        tools=[
            get_wisdom_filename_tool,
            read_wisdom_file_tool,
            execution_tracing_tool,
            write_wisdom_file_tool,
        ]
    )

    river_crossing_reflection_task.execute()


if __name__ == '__main__':
    plan_execute_reflect()
