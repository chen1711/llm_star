# 适配火山方舟DeepSeek模型的路径规划提示词配置
# 核心优化：简化格式要求，强化输出规范，提升解析成功率

sysprompt_parse = """You are a code generation assistant. Your task is to convert natural language descriptions of pathfinding problems into structured JSON objects that can be used as input for a pathfinding algorithm. The input will describe start and goal points, horizontal and vertical barriers, and the boundaries of the environment. 

For each input, extract the following information and format it into a JSON object:
1. "start": The starting point coordinates as [x, y].
2. "goal": The goal point coordinates as [x, y].
3. "range_x": The minimum and maximum x-coordinates of the environment range as [x_min, x_max].
4. "range_y": The minimum and maximum y-coordinates of the environment range as [y_min, y_max].
5. "horizontal_barriers": A list of horizontal barriers, each represented as [y, x_start, x_end].
6. "vertical_barriers": A list of vertical barriers, each represented as [x, y_start, y_end].

Ensure the JSON object is properly formatted and matches the required structure. ONLY return the JSON object, no other text."""

example_parse = {
    "sample1": 'find the shortest path from (5, 5) to (45, 25) on a 51x31 grid, avoiding horizontal barriers at [15, 10-21] and vertical barriers at [20, 0-15], [30, 15-30], and [40, 0-16].\n{"start": [5, 5], "goal": [45, 25], "range_x": [0, 51], "range_y": [0, 31], "horizontal_barriers": [[15, 10, 21]], "vertical_barriers": [[20, 0, 15], [30, 15, 30], [40, 0, 16]]}'
}

parse_llama = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a code generation assistant. Your task is to convert natural language descriptions of pathfinding problems into structured JSON objects that can be used as input for a pathfinding algorithm. The input will describe start and goal points, horizontal and vertical barriers, and the boundaries of the environment. 

For each input, extract the following information and format it into a JSON object:
1. "start": The starting point coordinates as [x, y].
2. "goal": The goal point coordinates as [x, y].
3. "range_x": The minimum and maximum x-coordinates of the environment range as [x_min, x_max].
4. "range_y": The minimum and maximum y-coordinates of the environment range as [y_min, y_max].
5. "horizontal_barriers": A list of horizontal barriers, each represented as [y, x_start, x_end].
6. "vertical_barriers": A list of vertical barriers, each represented as [x, y_start, y_end].

Ensure the JSON object is properly formatted, matches the required structure, and values are correctly extracted. (Do not return any comment, just the JSON)
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
find the shortest path from (5, 5) to (45, 25) on a 51x31 grid, avoiding horizontal barriers at [15, 10-21] and vertical barriers at [20, 0-15], [30, 15-30], and [40, 0-16].
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{"start": [5, 5], "goal": [45, 25], "range_x": [0, 51], "range_y": [0, 31], "horizontal_barriers": [[15, 10, 21]], "vertical_barriers": [[20, 0, 15], [30, 15, 30], [40, 0, 16]]}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
design a path from [29, 4] to [22, 21] on a 51 by 31 grid that avoids horizontal barriers centered at [[7, 21, 33]] and vertical barriers at [[16, 7, 31], [39, 3, 23], [1, 24, 24]].
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{"start": [29, 4], "goal": [22, 21], "range_x": [0, 51], "range_y": [0, 31], "horizontal_barriers": [[7, 21, 33]], "vertical_barriers": [[16, 7, 31], [39, 3, 23], [1, 24, 24]]}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
design a path from [44, 29] to [5, 23] on a 51 by 31 grid that avoids horizontal barriers centered at [[7, 21, 33]] and vertical barriers at [[16, 7, 31], [39, 3, 23], [1, 24, 24]].
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{"start": [44, 29], "goal": [5, 23], "range_x": [0, 51], "range_y": [0, 31], "horizontal_barriers": [[7, 21, 33]], "vertical_barriers": [[16, 7, 31], [39, 3, 23], [1, 24, 24]]}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{query}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

# 核心优化：简化输出要求，强制仅返回坐标列表，提升火山方舟模型解析成功率
standard_gpt = """
Find the shortest collision-free path between the start and goal points, avoiding all specified barriers.
Horizontal barriers: [y, x_start, x_end] (block all x from x_start to x_end at y)
Vertical barriers: [x, y_start, y_end] (block all y from y_start to y_end at x)

REQUIREMENTS:
1. Return ONLY the path coordinates as a list (e.g., [[5,5],[8,7],[12,9],[27,15]])
2. No explanations, no comments, no extra text
3. Coordinates must be within range x∈[0,51], y∈[0,31]
4. Path must be continuous (adjacent points differ by 1 in x or y)
5. Path must avoid all barriers

Start Point: {start}
Goal Point: {goal}
Horizontal Barriers: {horizontal_barriers}
Vertical Barriers: {vertical_barriers}
"""

standard_llama = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
Find the shortest collision-free path between the start and goal points, avoiding all specified barriers.
Horizontal barriers: [y, x_start, x_end] (block all x from x_start to x_end at y)
Vertical barriers: [x, y_start, y_end] (block all y from y_start to y_end at x)

REQUIREMENTS:
1. Return ONLY the path coordinates as a list (e.g., [[5,5],[8,7],[12,9],[27,15]])
2. No explanations, no comments, no extra text
3. Coordinates must be within range x∈[0,51], y∈[0,31]
4. Path must be continuous (adjacent points differ by 1 in x or y)
5. Path must avoid all barriers
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Start Point: {start}
Goal Point: {goal}
Horizontal Barriers: {horizontal_barriers}
Vertical Barriers: {vertical_barriers}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

cot_gpt = """
Find the shortest collision-free path between the start and goal points, avoiding all specified barriers.
Horizontal barriers: [y, x_start, x_end] (block all x from x_start to x_end at y)
Vertical barriers: [x, y_start, y_end] (block all y from y_start to y_end at x)

First, think step by step about how to navigate around the barriers:
1. Identify the direct path and blocked sections
2. Find the optimal detour around barriers
3. Ensure path continuity and shortness

THEN return ONLY the path coordinates as a list (e.g., [[5,5],[8,7],[12,9],[27,15]])
No explanations, no comments, no extra text.

Start Point: {start}
Goal Point: {goal}
Horizontal Barriers: {horizontal_barriers}
Vertical Barriers: {vertical_barriers}
"""

cot_llama = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
Find the shortest collision-free path between the start and goal points, avoiding all specified barriers.
Horizontal barriers: [y, x_start, x_end] (block all x from x_start to x_end at y)
Vertical barriers: [x, y_start, y_end] (block all y from y_start to y_end at x)

First, think step by step about how to navigate around the barriers:
1. Identify the direct path and blocked sections
2. Find the optimal detour around barriers
3. Ensure path continuity and shortness

THEN return ONLY the path coordinates as a list (e.g., [[5,5],[8,7],[12,9],[27,15]])
No explanations, no comments, no extra text.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Start Point: {start}
Goal Point: {goal}
Horizontal Barriers: {horizontal_barriers}
Vertical Barriers: {vertical_barriers}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

repe_gpt = """
Find the shortest collision-free path between the start and goal points, avoiding all specified barriers.
Use iterative refinement to generate the path:
1. Start at the initial point, select next point to avoid barriers
2. Evaluate each selected point (no collision, progress to goal)
3. Refine path until reaching goal

Horizontal barriers: [y, x_start, x_end] (block all x from x_start to x_end at y)
Vertical barriers: [x, y_start, y_end] (block all y from y_start to y_end at x)

REQUIREMENTS:
1. Return ONLY the final path coordinates as a list (e.g., [[5,5],[8,7],[12,9],[27,15]])
2. No iteration details, no explanations, no extra text
3. Coordinates must be within range x∈[0,51], y∈[0,31]
4. Path must be continuous and avoid all barriers

Start Point: {start}
Goal Point: {goal}
Horizontal Barriers: {horizontal_barriers}
Vertical Barriers: {vertical_barriers}
"""

repe_llama = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
Find the shortest collision-free path between the start and goal points, avoiding all specified barriers.
Use iterative refinement to generate the path:
1. Start at the initial point, select next point to avoid barriers
2. Evaluate each selected point (no collision, progress to goal)
3. Refine path until reaching goal

Horizontal barriers: [y, x_start, x_end] (block all x from x_start to x_end at y)
Vertical barriers: [x, y_start, y_end] (block all y from y_start to y_end at x)

REQUIREMENTS:
1. Return ONLY the final path coordinates as a list (e.g., [[5,5],[8,7],[12,9],[27,15]])
2. No iteration details, no explanations, no extra text
3. Coordinates must be within range x∈[0,51], y∈[0,31]
4. Path must be continuous and avoid all barriers
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Start Point: {start}
Goal Point: {goal}
Horizontal Barriers: {horizontal_barriers}
Vertical Barriers: {vertical_barriers}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

# 提示词映射表（保持原有接口，适配火山方舟）
gpt_prompt = {'standard': standard_gpt, 'cot': cot_gpt, 'repe': repe_gpt}
llama_prompt = {'standard': standard_llama, 'cot': cot_llama, 'repe': repe_llama}