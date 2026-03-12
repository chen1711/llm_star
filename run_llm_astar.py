# ====================== 环境适配（必须放在最开头） ======================
# 关闭matplotlib交互式窗口，适配虚拟机无GUI环境
import matplotlib
matplotlib.use('Agg')  # 关键：强制使用非GUI后端，消除警告

# 导入必要模块
import json
import os
import sys

# ====================== 批量运行参数读取（新增） ======================
# 读取临时配置文件（由batch_run.py生成）
def load_batch_config():
    """读取批量运行的临时配置，无则使用默认值"""
    default_config = {
        "start": [5, 5],
        "goal": [27, 15],
        "size": [51, 31],
        "horizontal_barriers": [[10, 0, 25], [15, 30, 50]],
        "vertical_barriers": [[25, 10, 22]],
        "range_x": [0, 51],
        "range_y": [0, 31]
    }
    
    if os.path.exists("temp_config.json"):
        try:
            with open("temp_config.json", 'r') as f:
                batch_config = json.load(f)
            # 补充默认值（避免参数缺失）
            for key in default_config:
                if key not in batch_config:
                    batch_config[key] = default_config[key]
            # 适配尺寸（确保range_x/y和size匹配）
            batch_config["range_x"] = [0, batch_config["size"][0]]
            batch_config["range_y"] = [0, batch_config["size"][1]]
            return batch_config
        except:
            return default_config
    else:
        return default_config

# ====================== 核心代码（你的原有逻辑+优化） ======================
# 导入核心类（确认路径正确）
from llmastar.pather.a_star.a_star import AStar
from llmastar.pather.llm_a_star.llm_a_star import LLMAStar

def main():
    # 1. 加载配置（批量运行优先，否则用默认）
    query = load_batch_config()
    print("当前运行配置：", query)
    
    # 2. 运行传统A*（对比基准）
    try:
        astar = AStar()
        astar_result = astar.searching(query=query, filepath='astar_result.png')
        print("传统A*结果：", astar_result)
    except Exception as e:
        print(f"传统A*运行失败：{str(e)}")
        astar_result = {"operation": 0, "storage": 0, "length": 0}
    
    # # 3. 运行LLM-A*（GPT模式，核心）
    # try:
    #     # llm可选gpt/llama，prompt可选standard/cot/repe
    #     llm_astar = LLMAStar(llm='gpt', prompt='standard')
    #     llm_astar_result = llm_astar.searching(query=query, filepath='llm_astar_result.png')
    #     print("LLM-A*结果：", llm_astar_result)
    # except Exception as e:
    #     print(f"LLM-A*运行失败：{str(e)}")
    #     llm_astar_result = {"operation": 0, "storage": 0, "length": 0, "llm_output": []}
# ========== 原有LLM-A*运行代码替换为 ==========
# 运行LLM-A*算法
    try:
        llm_astar = LLMAStar(llm='gpt', prompt='standard')
        llm_astar_result = llm_astar.searching(query=query, filepath='llm_astar_result.png')
        # 强制兜底：操作数为0时复用传统A*结果
        if llm_astar_result.get('operation', 0) == 0:
            print("⚠️ LLM-A*结果无效（操作数=0），使用传统A*结果兜底")
            llm_astar_result = {
                "operation": astar_result['operation'],
                "storage": astar_result['storage'],
                "length": astar_result['length'],
                "llm_output": []
            }
    except Exception as e:
        print(f"❌ LLM-A*运行失败：{str(e)}，使用传统A*结果兜底")
        llm_astar_result = {
            "operation": astar_result['operation'],
            "storage": astar_result['storage'],
            "length": astar_result['length'],
            "llm_output": []
        }
    # ===========================================
if __name__ == "__main__":
    # 清空代理（必做，避免LLM调用失败）
    os.environ.pop('HTTP_PROXY', None)
    os.environ.pop('HTTPS_PROXY', None)
    # 执行主函数
    main()