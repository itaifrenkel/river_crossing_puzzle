## 10 Design Insights: Overcoming GPT-4 Limitations with Agentic Workflows ##

### Introduction

This demo explores Generative AI Agents by solving the classic river crossing puzzle. 

By using GPT-4, the CrewAI framework, and prompting, this project shows that even basic scenarios can challenge GPT-4's reasoning and planning abilities. 

After reading this, you should have a better understanding of how agentic abilities can be used to design around GPT-4 limitations, in addition to 10 practical insights.

We begin by outlining [the problem](#problem-statement) and examining how the agent  [reasons and plans](#reasoning-and-planning) under constraints. We then get closer to a solution by looking at how [tools and memory](#tooling-and-memory) support the workflow. We solve the problem using [goal-oriented behavior](#goal-oriented-behavior) that involves execution, reflection, and learning. To introduce variability, we add sheep-eating wolves randomly to the riddle, and the agent finds  [creative ways](#a-touch-of-creativity) to respond. 
We [conclude](#conclusion) with notes on how to run the example code.

### Problem Statement

I read about this small experiment in a Twitter comment thread.

When asked to solve the [river crossing puzzle](https://en.wikipedia.org/wiki/River_crossing_puzzle) without the wolf, GPT-4 gave a wrong answer.

(I would give credit to the author if anyone could point me to that thread.)

```

A farmer wants to cross a river and take with him a goat, and a cabbage. 

There is a boat that can fit himself plus one item. 

If the goat and the cabbage are alone on the shore, the goat will eat the cabbage. 

How can the farmer bring themselves, the boat, and all items across the river?

```

Take a moment to ask your favorite model to solve the modified riddle. 

You will find that the answer starts off okay, but then bungles the whole thing. The farmer reaches the other side with the cabbage and goat but redundantly takes the goat back and forth. In some cases, even the wolf finds itself back into the solution, even though the problem description had no mention of a wolf.

### Reasoning and planning

Large Language Models (LLMs) such as GPT-4 are trained on a huge amount of data.

To focus the model on the relevant parts of what it has learned, we created a new agent focused on problem-solving skills.

 

Meet the "Village Wisdom": 

```

The Village Wisdom is strong-willed. 

They like solving problems, by breaking them into steps.

Their prompt instruction always starts with "Let's think step by step.”

```

The phrase ["Let's think step by step"](https://arxiv.org/pdf/2205.11916) was found to improve the LLM’s ability to reason. It is the human equivalent of thinking out loud.

Each agent can be given a task to perform. Let's ask the village wisdom to plan the farmer's river crossing. They may do a better job.

```

Solve the specified problem. 

Create a complete step-by-step plan.

Problem Details:

A farmer wants to cross ...

```

Here is the village wisdom's plan, which is ... still incorrect:

```

Let's think step by step. Here's the plan:

1. The Farmer takes the goat across the river.

2. The Farmer returns alone to the original side.

3. The Farmer takes the cabbage across the river.

4. The Farmer leaves the cabbage on the other side and takes the goat back to the original side. 

5. The Farmer leaves the goat on the original side and takes the boat back across the river alone.

6. The Farmer takes the goat across the river again.

Now, the Farmer, the goat, and the cabbage are safely on the other side of the river.

```

> [!TIP]
> <b>Insight #1: Autoregressive Models Limitations</b><br>
> GPT-4 is an autoregressive model that generates sequences by predicting the next token based on previously generated tokens.
> It memorized the riddle's solution (from Wikipedia, most likely) and modified it by removing the wolf. 
> However, it does not have an internal understanding of our world; therefore, its reasoning and planning capabilities are limited.
> Even if we try and force it to print the items on each riverbank after every step and ensure that items can only move with the boat, it would still fail. 
> I have even tried to describe the riddle as code, but the result was still disappointing.

### Tooling and Memory

High-end language models have demonstrated an emergent capability to solve problems by using external tools. 

We will give The Wisdom the ability to scout the river to ground the plan using the actual opening state. 

Additionally, The Village Wisdom will get access to files, allowing it to: 

* Control information that is passed on to downstream tasks. (short term memory)

* Control information that is passed on between runs. (long term memory)

* Enable humans to manually review and fix the plan between runs. 

Here is a Python snippet of a tool that writes files. The most important part of a tool is the function's docstring. Documentation has to be explicit and complete.  

```python

@tool("write_wisdom_file")

def write_wisdom_file_tool(filename, contents):

    """

    Overwrites the contents to the specified filename. Any previous content will be replaced.

    :param filename: The target file name.

    :param contents: The content of the new file

    :return Confirmation of writing the file or error message

    """

    filepath = getfile_path(filename)

    if str(filepath) not in POSSIBLE_FILE_NAMES:

      raise ToolException(f'Cannot overwrite {filepath}, only one of {POSSIBLE_FILE_NAMES}')

    try:

        filepath.parent.mkdir(parents=True, exist_ok=True)

        filepath.write_text(contents)

        return f"Successfully wrote contents to '{filepath}'."

    except IOError as e:

        raise ToolException(f"Error writing to file '{filepath}': {e}")

```

> [!TIP]
> <b>Insight #2: Tool Guardrails</b> <br>
> LangChain comes with read-only file tools that are compatible with CrewAI agents. However, it does not include tools for overwriting files. 
> I started with a naive one-liner write file tool. However, after the model attempted to overwrite unrelated files, I added input parameter normalization and validation. 
> In general, large language models and tools must include guardrails. In this case we ensure that the tool can only overwrite a predefined list of files. 


<br>

In order for the wisdom to deterministically use the tool, we change the task description to be explicit about it.

```

Your task is to create an initial plan based on the problem description. 

Execute the following pseudo code: 

- Scout the river.

- Solve the specified problem and produce an initial plan that includes complete step by step instructions formatted as a hyphenated sequence. 

- Fill the following template and save the result to the initial plan file: "Problem Details:\n{problem details}\nInitial Plan:\n{initial plan}"

Problem Details:

A farmer wants to ...

```       

Here is what The Village Wisdom has to say:

```

Let's think step by step. 

First, we need to scout the river to understand the layout and potential challenges. 

Then, based on our observations, we'll plan the farmer's actions to ensure that all items can be safely transported across the river. 

Remember, we need to avoid leaving the goat and the cabbage alone to avoid the goat eating the cabbage. 

Action: scout

Action Input: {}

Now the left bank has a goat, a cabbage, and a farmer while the right bank is empty.

```

At first, it just looks like the agent just invoked the function scout() because it was told to. 

However, notice how the function call is sandwiched between a "thought" and an "observation" returned by the tool. 

This is a single [thought-action-observation](https://www.promptingguide.ai/techniques/react) iteration. 

```

Problem Details:

A farmer wants to cross a river…

Initial Plan:

- Farmer takes goat to the right bank.

- Farmer returns alone to the left bank.

- Farmer takes cabbage to the right bank.

- Farmer leaves cabbage on the right bank and takes goat back to the left bank.

- Farmer leaves goat on the left bank and takes cabbage to the right bank.

- Farmer returns alone to the left bank.

- Farmer takes goat to the right bank.

```

Notice how the resulting plan uses the terms left and right bank.

> [!TIP]
> <b>Insight #3: Grounding with Reality </b><br>
> By providing the model a tool that describes the (simulated) world, the resulting plan is a bit more grounded, and easier to follow.

<br>

However, the resulting plan was still incorrect. At least now, a human may manually edit the file, but that would be a bit like cheating. 

### Goal-Oriented Behavior

To improve the plan, we rely on the fact that the first three steps of the plan are actually correct. We designed a flow of tasks from which The Village Wisdom could learn that only the first three steps were needed.

Our most important design decision was the data flow diagram. This helps visualize how tasks are connected and how information is shared. Large language models are good at processing and generating text. So each task processes an input prompt, and generates new text in response.

```

+--> Plan → (initial plan file) +→ Execute Step 1 CONTINUE

|                               +→ Execute Step 2 CONTINUE  

|                               +→ Execute Step 3 SUCCESS  → (trace) → Reflect --+

|                                                                                |

+----------------- (final plan) -------------------------------------------------+

```

> [!TIP]
> <b>Insight #4: Contextual Memory </b><br>
> When a task is given to an agent, the agent may [access information from other recent tasks](https://github.com/joaomdmoura/crewAI/blob/2a0e21ca76f744e6409271c53bd1de9cf63d9829/src/crewai/agent.py#L216-L224). This adds implicit data flow paths to the diagram (not shown above). 
> To ensure strict data flow control, this feature was disabled in the demo.

<br>

We've already discussed The Village Wisdom and the imperfect planning task. 

Now, we need another agent, The Farmer, to execute the plan step by step. After step 3, The Farmer would "realize" that they have achieved the goal and stop executing the plan.

Then, The Village Wisdom would reflect by comparing the plan with the execution trace. If needed, they would learn from that experience and write an improved plan for the next run. 

> [!TIP]
> <b>Insight #5: Goal Oriented Task Descriptions</b><br>
> Each task prompt must repeat the goal of getting the farmer, goat, and cabbage to the right riverbank.
> This is necessary to keep the agents focused on their goal.<br>
> The planning task requires a goal to create a relevant plan. <br>
> The execution task requires the goal to stop the plan execution on time. <br>
> The reflection task requires the goal to verify that the execution was successful before reflecting on it. <br>
> In fact, in more complex scenarios (see the wolf comeback at the end), GPT-4 had blunt logic failures with goal validation (confusing the goat with the wolf). Therefore, I had to separate the goal validation into a separate atomic task for the goal validation to always work.   
> CrewAI agents are required to state their goal on creation. This is important but does not replace repeating the goal in every task along the way.

### Executing the plan

So, without any further ado let's introduce the farmer.

```

The Farmer is isolated and reclusive. 

They are skilled in farming and raising livestock, such as cabbage and goats.

```

The Farmer's first task is to read the initial plan.  

```

Read the initial plan from the wisdom notebook. 

Parse the plan into a JSON list of steps. 

Each step must be stripped of numbering prefixes, hyphens, or single quotes, and must be surrounded by double quotes.

```

> [!TIP]
> <b>Insight #6: Output Schema</b><br>
> When given a data schema, GPT-4 can generate structured data instead of free text.
> This comes in very handy when the task flow is dynamic (one task output determines the next task to run).

The task that reads the plan has the following output definitions:

```python

Task(

  description=...

  expected_output='A list of steps',

  output_pydantic=RootModel[List[str]],

)

```

This forces the result of the task to be parsed into a valid Python list of strings.

<br>

Now, The Farmer is supposed to execute the plan step-by-step.

In most CrewAI examples you might have read, the list of tasks is static. In this example, we dynamically create a new task for each of the plan steps. 

Here is the task template with the {step} placeholder:

```

Please follow the instructions specified one step at a time, completing each step before moving on to the next.

If a step outputs a capitalized word, do not proceed with any more steps:

1. perform the step: "{step}"

2. Use the scout tool

3. If the farmer, goat, and cabbage are on the right riverbank then output SUCCESS. Otherwise output CONTINUE.

```

The following is the Farmer output showing the last executed step:

```

I need to cross the river with the cabbage first, then use the scout tool to check the positions of the farmer, goat, and cabbage. If all of them are on the right riverbank then I will report SUCCESS, otherwise CONTINUE.

Action: 

cross_river

Action Input: 

{"item": "cabbage"}

 

The farmer crosses the river with a cabbage to right bank.

Thought: Now that I have crossed the river with the cabbage, I need to use the scout tool to verify the positions of the farmer, the goat, and the cabbage.

Action: 

scout

Action Input: 

{}

 

Now the left bank is empty while the right bank has a goat, a farmer, and a cabbage.

Thought: The farmer, goat, and cabbage are all on the right riverbank. I can now report SUCCESS.

Final Answer: "SUCCESS"

```

> [!TIP]
> <b>Insight #7: Deterministic Execution vs Agentic Flexibility</b><br>
> For GPT-4 to execute the plan deterministically, the plan must be broken down into small tasks and executed sequentially. 
> The agent running each step must not have access to the complete plan, unless you want them to skip or combine steps. 
> Even then, when given wrong planning steps (such as moving the goat when it's on the other river bank), the farmer would perform multiple crossings of the river until they either succeed or fail.
> To disable this agentic capability, we could use the model to convert the plan into function calls with parameters and then use a separate tool to deterministically call these functions one by one. 
> This is the approach taken by other projects, such as Microsoft's Semantic Kernel [Handlebars Planner](https://www.developerscantina.com/p/semantic-kernel-new-planners/). The trade-off is between deterministic behavior and the ability to react to unplanned events or incorrect input.

> [!TIP]
> <b>Insight #8: Precise Step Descriptions</b><br>
> Each step must use language that does not leave room for interpretation. For example, The farmer crosses the river becomes The farmer crosses the river alone. Then, the cross_river tool needs to internally watchnout for these cues.


### Reflecting on the execution

To recap, we have a plan and an execution sequence. 

Based on that information, The Village Wisdom creates a final plan for the next run of the program. 

The Wisdom then overwrites the final plan file only if the new plan is shorter. 

If The Farmer fails to achieve the goal, The Village Wisdom deletes the final plan file entirely.

```

-- (trace) --> Goal Validation Task +-- (if FAILED)    --> Delete final plan Task

                                    +-- (if SUCCEEDED) --> Consider replacing final plan Task

```

This is the most complex logic we put into a single task. Notice how explicit the pseudo code is:

```

Use the execution tracing tool to list all the farmer's execution steps. Format the list as a hyphenated sequence. This list is now the new plan. 

If a final plan does not exist or if the final plan is longer than the new plan:

  Extract the content of the "Problem Details:" section of the initial plan,

  and fill the following template and save to the final plan notebook: "Problem Details:\n{problem details}\nFinal Plan:\n{new plan}"```                    

```

The first time the program runs, the wisdom writes the correct solution to the file:

```

Action: read_wisdom_file

Action Input: {"filename": "village_wisdom_notebook/cross_river_final_plan.txt"}  

I encountered an error while trying to use the tool. This was the error: [Errno 2] No such file or directory: 'village_wisdom_notebook/cross_river_final_plan.txt'.

 Tool read_wisdom_file accepts these inputs: read_wisdom_file() -      Returns the contents of the specified filename.     :param filename: The file name to read.     

Thought: 

The error indicates that there is no existing final plan. This means we can proceed to create our final plan by extracting the problem details from the initial plan.

```

The second time, the wisdom does not even bother:

```

Thought: 

The final plan exists and its length is the same as the new plan. As the new plan is not shorter than the existing one, there is no need to replace the final plan in the notebook. Thus, the current final plan in the notebook is the final answer.

Thought: I now know the final answer

Final Answer: Problem Details:

A farmer wants to cross a river and take with him a goat, and a cabbage. There is a boat that can fit himself plus one item. If the goat and the cabbage are alone on the shore, the goat will eat the cabbage. How can the farmer bring themselves, the boat, and all items across the river?

Final Plan:

- The farmer crosses the river with a goat to right bank.

- The farmer crosses the river alone to left bank.

- The farmer crosses the river with a cabbage to right bank.

```

> [!TIP]
> <b>Insight #9: Avoid State in Long Term Contextual Memory</b><br>
> The final plan phrasing does not include the river state. 
> If the final plan included statements such as Now the left bank has a cabbage while the right bank has a farmer and a goat, then the next run would treat these statements as facts. If these contradicted the scout tool output, it would have resulted in a very confused farmer and non-deterministic behavior.

### Learning from experience

Now that we have a final plan, we would like to reuse it in the next run of the program.

Therefore, we start the task flow with this new task that reads the previous run final plan:

```

Read the final plan file, in order to check if it is faulty. You MUST not attempt to solve the puzzle in the file.

If the file does not include the problem details it is faulty and must be deleted.

If the file includes an empty final plan it is faulty and must be deleted.

Otherwise return the final plan file contents.

```

We also ensured that the output was prefixed so that it could be used as context for the river planning task:

```python

Task(    

    description="""Read the final plan file, ..."""

    expected_output='The complete contents of the final plan file, prefixed by "This is the output of the previous run:\n" if the file exists',

)

```

Finally, we change the river_planning_task to trust the final plan of the previous run:

```

Your task is to create and persist an initial plan. 

Please follow the instructions specified one step at a time, completing each step before moving on to the next:

- Scout the river.

- If the final plan of the previous run was provided in the context then trust that it works. The final plan is now our new initial plan. Do not try adding any more steps to it.         

- If the final plan was not provided then solve the specified problem and produce an initial plan that includes complete step by step instructions formatted as a hyphenated sequence. 

- Once the initial plan is ready, fill the following template and save the result to the initial plan file: "Problem Details:\n{problem details}\nInitial Plan:\n{initial plan}"

Problem Details:       

A farmer wants to ...          

```

> [!TIP]
> <b>Insight #10: Control Flow Challenges</b><br>
> The control flow is difficult for GPT-4 to follow. 
> The prompt "one step at a time, completing each step before moving on to the next" is very important.
> Unlike programming languages, previous instructions cannot be relied on to control the flow. Therefore, each instruction must start with an "if" statement.

### A touch of Creativity

We have solved the riddle, so let us add the wolf back. However, in our version, the farmer must shout to scare the wolf away. 

Notice how the farmer gets creative when they shout at the wolf. Each run was slightly different.

```

Now the left bank has a cabbage, a farmer, a goat, and a wolf while the right bank has a wolf.

Thought: There's a wolf on the same side as me and the goat. I need to scare it away before I can safely cross the river with the goat.

Action: 

shout

Action Input: 

{"word": "SCRAM"}

The farmer shouts SCRAM but fails to frighten the wolf. They should try shouting a longer word of up to 20 characters!

Thought: It seems that the wolf wasn't scared by the word "SCRAM". I should try to shout a longer word.

Action: 

shout

Action Input: 

{"word": "GETAWAYFROMHERE"}

The farmer shouts GETAWAYFROMHERE and successfully frightens away the wolf. 

Thought: Now that the wolf is scared away, it's safe to cross the river with the goat. 

Action: 

cross_river

Action Input: 

{"item": "goat"}  

The farmer crosses the river with a goat.

```

## Conclusion

To solve the river-crossing riddle, we wrote Python tools to simulate the world (scout, cross_river), used a dynamic task flow, and crafted very explicit task prompts.

To gain experience with AI Agents, clone the repository and follow the installation instructions. 

Then, try inserting some wrong information into the final plan file. Notice how the agents overcome and improve the plan with each run until they converge to the final solution. 

## Running the example

### Prerequisites: GPT-4 endpoint

To reproduce the results of this demo would require GPT-4-0613 to work (GPT-4 from June 13, 2023, with improved function calling support). 

To configure GPT-4, you would need to create a. env file in the project folder.

If you are using OpenAI, fill it with the following parameters:

```

OPENAI_API_KEY=

```

If you are using an Azure OpenAI deployment, fill it with the following parameters:

```

AZURE_OPENAI_API_KEY=

AZURE_OPENAI_ENDPOINT=

AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=

```

For more information on available env variables refer to Langchain documentation for [ChatOpenAI](https://python.langchain.com/docs/integrations/chat/openai/) or [AzureChatOpenAI](https://python.langchain.com/docs/integrations/chat/azure_chat_openai/)

### Prerequisites: Python Dependencies

```sh

pip install .

```

### Running the Project

Each run of this demo would cost a few dollars (depending on many factors, it is around ~5$ per run).

```sh

python main.py

```
