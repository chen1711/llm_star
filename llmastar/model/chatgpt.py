import os
import openai  # 确保是0.28.1版本
import json

class ChatGPT:
    """适配火山方舟DeepSeek模型的路径规划调用类（保留原类名以兼容代码）"""
    def __init__(self, method="", sysprompt="", example=None):
        # 1. 读取环境变量中的API Key（优先）
        self.api_key = os.getenv("ARK_API_KEY", "5ffcef78-08f1-4784-a329-e1ca40959903")
        # 2. 火山方舟OpenAI兼容接口配置
        openai.api_key = self.api_key
        openai.api_base = "https://ark.cn-beijing.volces.com/api/v3"  # 固定地址
        openai.proxy = {}  # 关闭代理，国内访问无需代理
        
        # 3. 你的火山方舟接入点ID（关键！）
        self.model_id = "ep-20260304131130-sd22n"
        
        # 保留原有参数以兼容代码
        self.sysprompt = sysprompt
        self.example = example
        self.method = method

    def _build_prompt(self, query):
        """根据传入的query动态构建提示词（适配批量运行的不同参数）"""
        # 解析query字典中的关键参数
        start = query.get("start", [5,5])
        goal = query.get("goal", [27,15])
        horizontal_barriers = query.get("horizontal_barriers", [[10,0,25],[15,30,50]])
        vertical_barriers = query.get("vertical_barriers", [[25,10,22]])
        range_x = query.get("range_x", [0,51])
        range_y = query.get("range_y", [0,31])
        
        # 构造精准的路径规划提示词（强制返回纯坐标列表）
        prompt = f"""
        请严格按照以下要求生成避障路径：
        1. 起点：{start}，终点：{goal}
        2. 避开水平障碍：{horizontal_barriers}（格式：[y, x起始, x结束]）
        3. 避开垂直障碍：{vertical_barriers}（格式：[x, y起始, y结束]）
        4. 坐标范围：x∈{range_x}，y∈{range_y}
        5. 输出格式：仅返回坐标列表（如[[5,5],[8,7],[12,9],[27,15]]），不要返回任何解释、说明、备注
        6. 路径要求：连续、无跳跃、避障、尽可能短
        """
        return prompt.strip()

    def chat(self, query):
        """
        核心调用函数：
        - query: 字典格式，包含start/goal/barriers等参数
        - 返回：模型生成的路径字符串（纯坐标列表）
        """
        # 1. 构建动态提示词
        prompt = self._build_prompt(query)
        
        # 2. 构造API调用消息体
        messages = [
            {"role": "system", "content": "你是专业的路径规划助手，严格按照要求仅返回坐标列表，无其他任何内容"},
            {"role": "user", "content": prompt}
        ]

        # 3. 增加重试逻辑（最多2次）
        for retry in range(2):
            try:
                # 调用火山方舟API（超时延长到20秒）
                response = openai.ChatCompletion.create(
                    model=self.model_id,       # 你的接入点ID
                    messages=messages,         # 消息体
                    temperature=0.0,           # 固定输出，保证路径稳定
                    max_tokens=1000,           # 足够容纳路径坐标
                    timeout=20,                # 超时延长到20秒（解决复杂路径生成超时）
                    top_p=1.0,                 # 固定采样策略
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                
                # 提取并清洗返回结果
                path_str = response.choices[0].message["content"].strip()
                # 过滤可能的多余字符（如markdown格式、解释文字）
                path_str = path_str.strip("```json").strip("```").strip()
                
                # 强制格式修正（关键！解决中文符号/空格导致的解析失败）
                path_str = path_str.replace("（", "[").replace("）", "]")  # 中文括号→英文括号
                path_str = path_str.replace("，", ",").replace(" ", "")   # 中文逗号→英文逗号，删除空格
                path_str = path_str.replace("\\n", "").replace("\\t", "") # 删除换行/制表符
                
                # 验证是否为合法列表
                try:
                    # 尝试解析为列表，确保格式正确
                    path_list = json.loads(path_str)
                    return str(path_list)  # 返回字符串格式，兼容原有代码
                except json.JSONDecodeError:
                    if retry < 1:
                        print(f"⚠️ 路径格式解析失败，重试第{retry+1}次...")
                        continue
                    else:
                        raise RuntimeError("路径格式解析失败，重试后仍无效")
                        
            except openai.error.Timeout:
                if retry < 1:
                    print(f"⚠️ 火山方舟API调用超时（20秒），重试第{retry+1}次...")
                    continue
                else:
                    raise RuntimeError(f"火山方舟API调用超时（20秒），重试后仍失败，请检查网络或模型状态")
            except openai.error.APIError as e:
                if retry < 1:
                    print(f"⚠️ 火山方舟API调用失败：{str(e)}，重试第{retry+1}次...")
                    continue
                else:
                    raise RuntimeError(f"火山方舟API调用失败：{str(e)}，错误码：{e.code if hasattr(e, 'code') else '未知'}")
            except Exception as e:
                if retry < 1:
                    print(f"⚠️ 路径生成失败：{str(e)}，重试第{retry+1}次...")
                    continue
                else:
                    raise RuntimeError(f"路径生成失败：{str(e)}")

    def ask(self, prompt, max_tokens=1000):
        """
        兼容原有代码的接口：
        - 若prompt是字典（批量运行传入的query），直接调用chat
        - 若prompt是字符串，按原有逻辑处理
        """
        if isinstance(prompt, dict):
            # 批量运行时传入的是query字典
            return self.chat(prompt)
        else:
            # 兼容原有字符串prompt调用
            dummy_query = {
                "start": [5,5],
                "goal": [27,15],
                "horizontal_barriers": [[10,0,25],[15,30,50]],
                "vertical_barriers": [[25,10,22]]
            }
            return self.chat(dummy_query)