import json
import os
import time
import pandas as pd
import subprocess
import re
import ast
import numpy as np

# ====================== 配置项 ======================
JSON_PATH = "dataset/environment_50_30.json"
RESULT_CSV = "llm_astar_batch_results.csv"
RUN_SCRIPT = "run_llm_astar.py"
ARK_API_KEY = "5ffcef78-08f1-4784-a329-e1ca40959903"

# 测试配置
TEST_TOTAL_COUNT = 100  # 100组测试
TEST_MAP_IDS = None     # 遍历所有地图

# 清空代理
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ['ARK_API_KEY'] = ARK_API_KEY

# ====================== 核心解析函数（保留） ======================
def parse_script_output(output):
    result = {
        "astar_operation": 0,
        "astar_storage": 0,
        "astar_length": 0,
        "llmastar_operation": 0,
        "llmastar_storage": 0,
        "llmastar_length": 0,
        "original_llmastar_operation": 0,
        "original_llmastar_storage": 0,
        "original_llmastar_length": 0
    }
    
    # 匹配传统A*
    astar_pattern = r"\{'operation': (\d+), 'storage': (\d+), 'length': ([\d\.]+)\}"
    astar_lines = []
    for line in output.split('\n'):
        if '传统A*结果' in line or ('operation' in line and 'storage' in line and 'length' in line and 'llm_output' not in line):
            astar_lines.append(line)
    
    for line in astar_lines:
        astar_match = re.search(astar_pattern, line)
        if astar_match:
            result['astar_operation'] = int(astar_match.group(1))
            result['astar_storage'] = int(astar_match.group(2))
            result['astar_length'] = float(astar_match.group(3))
            break
    
    # 匹配LLM-A*
    llm_pattern = r"\{'operation': (\d+), 'storage': (\d+), 'length': ([\d\.]+), 'llm_output': .*\}"
    llm_lines = [line for line in output.split('\n') if 'llm_output' in line and 'operation' in line]
    
    for line in llm_lines:
        llm_match = re.search(llm_pattern, line)
        if llm_match:
            result['original_llmastar_operation'] = int(llm_match.group(1))
            result['original_llmastar_storage'] = int(llm_match.group(2))
            result['original_llmastar_length'] = float(llm_match.group(3))
            result['llmastar_operation'] = int(llm_match.group(1))
            result['llmastar_storage'] = int(llm_match.group(2))
            result['llmastar_length'] = float(llm_match.group(3))
            break
    
    return result

# ====================== 单组运行函数（保留） ======================
def run_single_experiment(map_data, sg_idx, current_total_idx):
    sg_item = map_data['start_goal'][sg_idx]
    start = sg_item[0] if len(sg_item)>=2 else [5,5]
    goal = sg_item[1] if len(sg_item)>=2 else [27,15]
    x_size = map_data['range_x'][1] - 1
    y_size = map_data['range_y'][1] - 1
    horizontal_barriers = map_data['horizontal_barriers']
    vertical_barriers = map_data['vertical_barriers']
    
    # 临时配置文件
    temp_config = {
        "start": start,
        "goal": goal,
        "size": [x_size, y_size],
        "horizontal_barriers": horizontal_barriers,
        "vertical_barriers": vertical_barriers
    }
    with open("temp_config.json", 'w') as f:
        json.dump(temp_config, f)
    
    try:
        print(f"\n=== 运行第{current_total_idx+1}/{TEST_TOTAL_COUNT}组（地图{map_data['id']}-样本{sg_idx+1}）===")
        print(f"起点: {start} | 终点: {goal}")
        
        # 执行脚本
        completed_process = subprocess.run(
            ['python3', RUN_SCRIPT],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # 解析输出
        output = completed_process.stdout + completed_process.stderr
        parsed_result = parse_script_output(output)
        
        # 兜底逻辑
        fallback_flag = False
        if parsed_result['llmastar_operation'] == 0 and parsed_result['astar_operation'] > 0:
            original_op = parsed_result['llmastar_operation']
            parsed_result['llmastar_operation'] = parsed_result['astar_operation']
            parsed_result['llmastar_storage'] = parsed_result['astar_storage']
            parsed_result['llmastar_length'] = parsed_result['astar_length']
            fallback_flag = True
            print(f"⚠️ 第{current_total_idx+1}组：LLM-A*操作数={original_op}（无效），已替换为传统A*数值")
        
        # 打印单组完整结果
        print(f"传统A* - 操作数: {parsed_result['astar_operation']} | 存储数: {parsed_result['astar_storage']} | 路径长度: {parsed_result['astar_length']:.4f}")
        print(f"LLM-A*  - 操作数: {parsed_result['llmastar_operation']} | 存储数: {parsed_result['llmastar_storage']} | 路径长度: {parsed_result['llmastar_length']:.4f}")
        if fallback_flag:
            print(f"原始LLM-A* - 操作数: {parsed_result['original_llmastar_operation']} | 存储数: {parsed_result['original_llmastar_storage']} | 路径长度: {parsed_result['original_llmastar_length']:.4f}")
        
        # 整理结果
        final_result = {
            "map_id": map_data['id'],
            "sg_idx": sg_idx,
            "total_idx": current_total_idx,
            "start": str(start),
            "goal": str(goal),
            "astar_operation": parsed_result['astar_operation'],
            "astar_storage": parsed_result['astar_storage'],
            "astar_length": parsed_result['astar_length'],
            "llmastar_operation": parsed_result['llmastar_operation'],
            "llmastar_storage": parsed_result['llmastar_storage'],
            "llmastar_length": parsed_result['llmastar_length'],
            "original_llmastar_operation": parsed_result['original_llmastar_operation'],
            "original_llmastar_storage": parsed_result['original_llmastar_storage'],
            "original_llmastar_length": parsed_result['original_llmastar_length'],
            "fallback": fallback_flag,
            # 新增：计算单组比率
            "operation_ratio": parsed_result['llmastar_operation'] / parsed_result['astar_operation'] if parsed_result['astar_operation']>0 else 0,
            "storage_ratio": parsed_result['llmastar_storage'] / parsed_result['astar_storage'] if parsed_result['astar_storage']>0 else 0,
            "length_ratio": parsed_result['llmastar_length'] / parsed_result['astar_length'] if parsed_result['astar_length']>0 else 0
        }
        
        return final_result
        
    except subprocess.TimeoutExpired:
        print(f"❌ 第{current_total_idx+1}组超时！")
        return None
    except Exception as e:
        print(f"❌ 第{current_total_idx+1}组出错：{str(e)}")
        return None
    finally:
        if os.path.exists("temp_config.json"):
            os.remove("temp_config.json")

# ====================== 批量运行主函数（新增全维度统计） ======================
def batch_run():
    # 检查文件
    if not os.path.exists(JSON_PATH):
        print(f"❌ 错误：数据集文件 {JSON_PATH} 不存在！")
        return
    if not os.path.exists(RUN_SCRIPT):
        print(f"❌ 错误：运行脚本 {RUN_SCRIPT} 不存在！")
        return
    
    # 读取数据集
    with open(JSON_PATH, 'r') as f:
        maps_data = json.load(f)
    
    # 筛选地图
    if TEST_MAP_IDS is not None:
        maps_data = [m for m in maps_data if m['id'] in TEST_MAP_IDS]
    if len(maps_data) == 0:
        print(f"❌ 没有找到符合条件的地图！")
        return
    
    # 初始化统计
    all_results = []
    completed = 0
    fallback_count = 0
    current_total_idx = 0
    
    print(f"\n🚀 开始测试总计{TEST_TOTAL_COUNT}组样本（涉及{len(maps_data)}张地图）...")
    
    # 运行100组
    for map_data in maps_data:
        if current_total_idx >= TEST_TOTAL_COUNT:
            break
        sg_count = len(map_data['start_goal'])
        for sg_idx in range(sg_count):
            if current_total_idx >= TEST_TOTAL_COUNT:
                break
            result = run_single_experiment(map_data, sg_idx, current_total_idx)
            if result:
                all_results.append(result)
                completed += 1
                if result['fallback']:
                    fallback_count += 1
            current_total_idx += 1
    
    # 保存完整结果到CSV
    df = pd.DataFrame(all_results)
    df.to_csv(RESULT_CSV, index=False, encoding='utf-8-sig')  # 兼容Excel
    
    # ========== 新增：全维度统计输出 ==========
    print("\n" + "="*50)
    print("✅ 100组批量测试完整统计报告")
    print("="*50)
    
    # 1. 基础完成情况
    print(f"\n📋 测试基础信息")
    print(f"   计划测试组数: {TEST_TOTAL_COUNT}")
    print(f"   实际完成组数: {completed}/{current_total_idx}")
    print(f"   完成率: {completed/current_total_idx*100:.1f}%" if current_total_idx>0 else "0%")
    print(f"   LLM-A*兜底组数: {fallback_count}/{completed} (失败率: {fallback_count/completed*100:.1f}%)" if completed>0 else "0%")
    
    if completed > 0:
        # 2. 核心指标均值统计
        print(f"\n📊 核心指标均值对比（传统A* vs LLM-A*）")
        print(f"   ┌───────────────┬───────────┬───────────┬────────────┐")
        print(f"   │ 指标          │ 传统A*    │ LLM-A*    │ 比率(LLM/A*)│")
        print(f"   ├───────────────┼───────────┼───────────┼────────────┤")
        print(f"   │ 平均操作数    │ {df['astar_operation'].mean():7.2f} │ {df['llmastar_operation'].mean():7.2f} │ {df['operation_ratio'].mean():9.2f} │")
        print(f"   │ 平均存储数    │ {df['astar_storage'].mean():7.2f} │ {df['llmastar_storage'].mean():7.2f} │ {df['storage_ratio'].mean():9.2f} │")
        print(f"   │ 平均路径长度  │ {df['astar_length'].mean():7.4f} │ {df['llmastar_length'].mean():7.4f} │ {df['length_ratio'].mean():9.2f} │")
        print(f"   └───────────────┴───────────┴───────────┴────────────┘")
        
        # 3. 原始LLM-A*数据（未兜底）
        print(f"\n📈 原始LLM-A*数据统计（未兜底）")
        original_ops = df['original_llmastar_operation']
        original_storages = df['original_llmastar_storage']
        original_lengths = df['original_llmastar_length']
        
        # 过滤掉0值（失败样本）
        valid_original_ops = original_ops[original_ops > 0]
        valid_original_storages = original_storages[original_storages > 0]
        valid_original_lengths = original_lengths[original_lengths > 0]
        
        print(f"   原始平均操作数 (有效样本): {valid_original_ops.mean():.2f} (共{len(valid_original_ops)}个有效样本)")
        print(f"   原始平均存储数 (有效样本): {valid_original_storages.mean():.2f}")
        print(f"   原始平均路径长度 (有效样本): {valid_original_lengths.mean():.4f}")
        print(f"   原始失败率: {len(original_ops[original_ops==0])/completed*100:.1f}%")
        
        # 4. 按地图详细统计
        print(f"\n🗺️  按地图维度详细统计")
        map_stats = df.groupby('map_id').agg({
            'astar_operation': ['mean', 'std', 'min', 'max'],
            'llmastar_operation': ['mean', 'std', 'min', 'max'],
            'original_llmastar_operation': lambda x: (x == 0).sum(),
            'fallback': 'count'
        }).round(2)
        
        # 重命名列
        map_stats.columns = ['传统A*操作数-均值', '传统A*操作数-标准差', '传统A*操作数-最小值', '传统A*操作数-最大值',
                             'LLM-A*操作数-均值', 'LLM-A*操作数-标准差', 'LLM-A*操作数-最小值', 'LLM-A*操作数-最大值',
                             'LLM-A*失败组数', '总组数']
        map_stats['LLM-A*失败率(%)'] = (map_stats['LLM-A*失败组数'] / map_stats['总组数'] * 100).round(1)
        
        # 打印按地图统计表格
        print(map_stats)
        
        # 5. 路径长度分布
        print(f"\n📏 路径长度分布统计")
        length_diff = df['llmastar_length'] - df['astar_length']
        print(f"   LLM-A*与传统A*路径长度差值均值: {length_diff.mean():.4f}")
        print(f"   路径长度差值标准差: {length_diff.std():.4f}")
        print(f"   路径长度最优样本数 (LLM-A* ≤ 传统A*): {len(length_diff[length_diff <= 0])}")
        print(f"   路径长度最优率: {len(length_diff[length_diff <= 0])/completed*100:.1f}%")
        
        # 6. 资源消耗分析
        print(f"\n💻 资源消耗分析")
        op_increase = (df['llmastar_operation'] - df['astar_operation']).mean()
        storage_increase = (df['llmastar_storage'] - df['astar_storage']).mean()
        print(f"   LLM-A*平均操作数增加量: {op_increase:.2f} (+{op_increase/df['astar_operation'].mean()*100:.1f}%)")
        print(f"   LLM-A*平均存储数增加量: {storage_increase:.2f} (+{storage_increase/df['astar_storage'].mean()*100:.1f}%)")
    
    # 7. 保存详细统计报告
    with open("llm_astar_100group_report.txt", 'w', encoding='utf-8') as f:
        f.write("LLM-A* 100组测试详细报告\n")
        f.write("="*60 + "\n\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {(time.time() - start_time)/60:.1f}分钟\n\n")
        f.write(f"基础信息:\n")
        f.write(f"  计划测试组数: {TEST_TOTAL_COUNT}\n")
        f.write(f"  实际完成组数: {completed}/{current_total_idx}\n")
        f.write(f"  LLM-A*兜底失败率: {fallback_count/completed*100:.1f}%\n\n")
        
        if completed > 0:
            f.write("核心指标均值:\n")
            f.write(f"  传统A*平均操作数: {df['astar_operation'].mean():.2f}\n")
            f.write(f"  LLM-A*平均操作数: {df['llmastar_operation'].mean():.2f}\n")
            f.write(f"  平均操作数比率: {df['operation_ratio'].mean():.2f}\n")
            f.write(f"  平均路径长度比率: {df['length_ratio'].mean():.2f}\n\n")
            
            f.write("按地图统计:\n")
            f.write(map_stats.to_string())
    
    print(f"\n📁 输出文件说明")
    print(f"   1. 详细结果数据: {RESULT_CSV} (Excel可直接打开)")
    print(f"   2. 完整统计报告: llm_astar_100group_report.txt")

# ====================== 运行入口 ======================
if __name__ == "__main__":
    start_time = time.time()
    batch_run()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n⏱️ 测试总耗时: {elapsed/3600:.1f}小时 ({elapsed/60:.1f}分钟)")