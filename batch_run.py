import json
import os
import time
import pandas as pd
import subprocess
import re
import ast

# ====================== 配置项（根据你的实际情况修改） ======================
JSON_PATH = "dataset/environment_50_30.json"  # 数据集JSON路径
RESULT_CSV = "llm_astar_batch_results.csv"     # 结果保存文件
RUN_SCRIPT = "run_llm_astar.py"                # 你的核心脚本
ARK_API_KEY = "5ffcef78-08f1-4784-a329-e1ca40959903"  # 你的API Key

# 清空代理（必做）
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ['ARK_API_KEY'] = ARK_API_KEY

# ====================== 核心解析函数（修复后：适配实际输出格式） ======================
def parse_script_output(output):
    """解析run_llm_astar.py的终端输出，提取关键指标（适配无中文前缀的纯字典格式）"""
    result = {
        "astar_operation": 0,
        "astar_storage": 0,
        "astar_length": 0,
        "llmastar_operation": 0,
        "llmastar_storage": 0,
        "llmastar_length": 0,
        "original_llmastar_operation": 0,  # 备份原始LLM-A*操作数
        "original_llmastar_storage": 0,    # 备份原始LLM-A*存储数
        "original_llmastar_length": 0      # 备份原始LLM-A*路径长度
    }
    
    # ========== 1. 匹配传统A*结果（兼容有/无中文前缀） ==========
    # 匹配格式：{'operation': 218, 'storage': 237, 'length': 28.485...}
    astar_pattern = r"\{'operation': (\d+), 'storage': (\d+), 'length': ([\d\.]+)\}"
    
    # 先找包含传统A*结果的行（优先匹配带前缀的，再匹配纯字典）
    astar_lines = []
    for line in output.split('\n'):
        if '传统A*结果' in line or ('operation' in line and 'storage' in line and 'length' in line and 'llm_output' not in line):
            astar_lines.append(line)
    
    # 解析传统A*结果
    for line in astar_lines:
        astar_match = re.search(astar_pattern, line)
        if astar_match:
            result['astar_operation'] = int(astar_match.group(1))
            result['astar_storage'] = int(astar_match.group(2))
            result['astar_length'] = float(astar_match.group(3))
            break
    
    # ========== 2. 匹配LLM-A*结果（带llm_output的字典） ==========
    # 匹配格式：{'operation': 212, 'storage': 217, 'length': 28.485..., 'llm_output': [...]}
    llm_pattern = r"\{'operation': (\d+), 'storage': (\d+), 'length': ([\d\.]+), 'llm_output': .*\}"
    
    # 找包含LLM-A*结果的行（带llm_output的字典）
    llm_lines = [line for line in output.split('\n') if 'llm_output' in line and 'operation' in line]
    
    # 解析LLM-A*结果
    for line in llm_lines:
        llm_match = re.search(llm_pattern, line)
        if llm_match:
            # 备份原始数据
            result['original_llmastar_operation'] = int(llm_match.group(1))
            result['original_llmastar_storage'] = int(llm_match.group(2))
            result['original_llmastar_length'] = float(llm_match.group(3))
            # 赋值给LLM-A*结果
            result['llmastar_operation'] = int(llm_match.group(1))
            result['llmastar_storage'] = int(llm_match.group(2))
            result['llmastar_length'] = float(llm_match.group(3))
            break
    
    return result

# ====================== 单组实验运行函数（新增兜底逻辑） ======================
def run_single_experiment(map_data, sg_idx):
    """运行单组实验：传递参数 → 执行脚本 → 解析结果 → 兜底处理"""
    # 1. 解析当前样本的参数
    sg_item = map_data['start_goal'][sg_idx]
    start = sg_item[0] if len(sg_item)>=2 else [5,5]
    goal = sg_item[1] if len(sg_item)>=2 else [27,15]
    x_size = map_data['range_x'][1] - 1  # 适配你的51→50
    y_size = map_data['range_y'][1] - 1  # 适配你的31→30
    horizontal_barriers = map_data['horizontal_barriers']
    vertical_barriers = map_data['vertical_barriers']
    
    # 2. 构造临时配置（写入临时文件，避免环境变量传递复杂数据）
    temp_config = {
        "start": start,
        "goal": goal,
        "size": [x_size, y_size],
        "horizontal_barriers": horizontal_barriers,
        "vertical_barriers": vertical_barriers
    }
    with open("temp_config.json", 'w') as f:
        json.dump(temp_config, f)
    
    # 3. 执行run_llm_astar.py
    try:
        print(f"\n=== 运行第{sg_idx+1}/10组 ===")
        print(f"起点: {start} | 终点: {goal}")
        # 执行脚本并捕获输出
        completed_process = subprocess.run(
            ['python3', RUN_SCRIPT],
            capture_output=True,
            text=True,
            timeout=300  # 超时5分钟
        )
        
        # 4. 解析输出
        output = completed_process.stdout + completed_process.stderr
        parsed_result = parse_script_output(output)
        
        # ========== 新增：LLM-A*操作数=0兜底逻辑 ==========
        fallback_flag = False
        if parsed_result['llmastar_operation'] == 0 and parsed_result['astar_operation'] > 0:
            # 备份原始无效数据
            original_op = parsed_result['llmastar_operation']
            # 替换为传统A*的有效数据
            parsed_result['llmastar_operation'] = parsed_result['astar_operation']
            parsed_result['llmastar_storage'] = parsed_result['astar_storage']
            parsed_result['llmastar_length'] = parsed_result['astar_length']
            # 标记兜底并打印日志
            fallback_flag = True
            print(f"⚠️ 批量测试-第{sg_idx+1}组：LLM-A*操作数={original_op}（无效），已替换为传统A*数值(操作数={parsed_result['astar_operation']})")
        
        # 5. 打印单组结果（直观查看）
        print(f"传统A* - 操作数: {parsed_result['astar_operation']} | 存储数: {parsed_result['astar_storage']} | 路径长度: {parsed_result['astar_length']}")
        print(f"LLM-A* - 操作数: {parsed_result['llmastar_operation']} | 存储数: {parsed_result['llmastar_storage']} | 路径长度: {parsed_result['llmastar_length']}")
        
        # 6. 整理结果
        final_result = {
            "map_id": map_data['id'],
            "sg_idx": sg_idx,
            "start": str(start),
            "goal": str(goal),
            "astar_operation": parsed_result['astar_operation'],
            "astar_storage": parsed_result['astar_storage'],
            "astar_length": parsed_result['astar_length'],
            "llmastar_operation": parsed_result['llmastar_operation'],
            "llmastar_storage": parsed_result['llmastar_storage'],
            "llmastar_length": parsed_result['llmastar_length'],
            # 备份原始LLM-A*数据（便于后续分析）
            "original_llmastar_operation": parsed_result['original_llmastar_operation'],
            "original_llmastar_storage": parsed_result['original_llmastar_storage'],
            "original_llmastar_length": parsed_result['original_llmastar_length'],
            # 标记是否兜底
            "fallback": fallback_flag
        }
        
        # 7. 计算效率比率（避免除零错误）
        final_result['operation_ratio'] = parsed_result['llmastar_operation'] / parsed_result['astar_operation'] if parsed_result['astar_operation']>0 else 0
        final_result['storage_ratio'] = parsed_result['llmastar_storage'] / parsed_result['astar_storage'] if parsed_result['astar_storage']>0 else 0
        final_result['length_ratio'] = parsed_result['llmastar_length'] / parsed_result['astar_length'] if parsed_result['astar_length']>0 else 0
        
        return final_result
        
    except subprocess.TimeoutExpired:
        print(f"❌ 第{sg_idx+1}组超时！")
        return None
    except Exception as e:
        print(f"❌ 第{sg_idx+1}组出错：{str(e)}")
        return None
    finally:
        # 删除临时文件
        if os.path.exists("temp_config.json"):
            os.remove("temp_config.json")

# ====================== 批量运行主函数（新增统计） ======================
def batch_run():
    # 1. 检查必要文件
    if not os.path.exists(JSON_PATH):
        print(f"❌ 错误：数据集文件 {JSON_PATH} 不存在！")
        return
    if not os.path.exists(RUN_SCRIPT):
        print(f"❌ 错误：运行脚本 {RUN_SCRIPT} 不存在！")
        return
    
    # 2. 读取数据集
    with open(JSON_PATH, 'r') as f:
        maps_data = json.load(f)
    
    # 3. 初始化结果列表和统计变量
    all_results = []
    completed = 0
    fallback_count = 0  # 统计兜底次数
    test_sg_count = 10  # 只测试10组
    
    # 4. 仅运行第0张地图的前10组样本
    test_map_id = 0
    target_map = None
    for map_data in maps_data:
        if map_data['id'] == test_map_id:
            target_map = map_data
            break
    
    if target_map is None:
        print(f"❌ 找不到ID为{test_map_id}的地图！")
        return
    
    print(f"\n🚀 开始测试第{test_map_id}张地图的{test_sg_count}组样本...")
    
    # 运行10组起止点
    for sg_idx in range(min(test_sg_count, len(target_map['start_goal']))):
        result = run_single_experiment(target_map, sg_idx)
        if result:
            all_results.append(result)
            completed += 1
            # 统计兜底次数
            if result['fallback']:
                fallback_count += 1
    
    # 5. 保存结果到CSV
    df = pd.DataFrame(all_results)
    df.to_csv(RESULT_CSV, index=False, encoding='utf-8')
    
    # 6. 输出测试报告（新增兜底统计）
    print("\n✅ 10组样本测试完成！")
    print(f"📊 有效实验数: {completed}/{test_sg_count}")
    if completed > 0:
        fallback_rate = fallback_count / completed * 100
        print(f"⚠️ LLM-A*兜底组数: {fallback_count}/{completed} (失败率: {fallback_rate:.1f}%)")
    print(f"📁 结果文件: {RESULT_CSV}")
    
    # 7. 输出统计结果
    print("\n=== 10组样本统计结果 ===")
    if completed > 0:
        print(f"传统A*平均操作数: {df['astar_operation'].mean():.2f}")
        print(f"LLM-A*平均操作数: {df['llmastar_operation'].mean():.2f}")
        print(f"平均操作数比率 (LLM-A*/A*): {df['operation_ratio'].mean():.2f}")
        print(f"平均路径长度比率: {df['length_ratio'].mean():.2f}")
        
        # 输出原始LLM-A*数据统计（未兜底前）
        print(f"\n=== 原始LLM-A*数据统计（未兜底） ===")
        original_op_mean = df['original_llmastar_operation'].mean()
        original_fail_count = len(df[df['original_llmastar_operation']==0])
        original_fail_rate = original_fail_count / completed * 100 if completed > 0 else 0
        print(f"原始LLM-A*平均操作数: {original_op_mean:.2f}")
        print(f"原始LLM-A*失败率: {original_fail_rate:.1f}%")

# ====================== 运行入口 ======================
if __name__ == "__main__":
    start_time = time.time()
    batch_run()
    end_time = time.time()
    print(f"\n⏱️ 测试总耗时: {(end_time - start_time)/60:.2f} 分钟")