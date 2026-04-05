import os
import csv
import numpy as np
from typing import List, Dict, Any

class DialecticalPhysiotherapy:
    def __init__(self, csv_path: str):
        """
        中医辨证理疗方案类
        - 参数:
            - csv_path: 辨证方案CSV文件路径
        """
        self.csv_path = csv_path
        self.dialectical_map = self._load_dialectical_map()
    
    def _load_dialectical_map(self) -> Dict[str, Dict[str, str]]:
        """
        加载辨证方案映射
        - 返回:
            - Dict[str, Dict[str, str]]: 6位编码到理疗方案的映射
        """
        dialectical_map = {}
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    code = row['6 位编码']
                    if len(code) == 6:
                        dialectical_map[code] = {
                            '患者表征': row['患者表征'],
                            '针灸治疗': row['针灸治疗'],
                            '中药选择': row['中药选择'],
                            '食疗方案': row['食疗方案']
                        }
        except Exception as e:
            print(f"加载辨证方案失败: {e}")
        return dialectical_map
    
    def generate_dialectical_code(self, classification_results: List[Dict[str, Any]]) -> str:
        """
        根据分类结果生成辨证编码
        - 参数:
            - classification_results: 分类结果列表
        - 返回:
            - str: 6位辨证编码
        """
        if not classification_results:
            return "000000"
        
        # 初始化最大标签值
        max_labels = {
            '分叶': 0,
            '棘刺': 0,
            '纹理': 0,
            '边缘': 0,
            '钙化': 0,
            '恶性程度': 0
        }
        
        # 任务名称映射（处理可能的命名差异）
        task_mapping = {
            'lobulation': '分叶',
            'spiculation': '棘刺',
            'texture': '纹理',
            'margin': '边缘',
            'calcification': '钙化',
            'malignancy': '恶性程度'
        }
        
        # 遍历所有结节，取最大标签值
        for result in classification_results:
            # 提取各特征的标签值
            for task in result:
                # 尝试映射任务名称
                mapped_task = task_mapping.get(task, task)
                if mapped_task in max_labels:
                    try:
                        pred_label = result[task].get('pred_label', 0)
                        if pred_label > max_labels[mapped_task]:
                            max_labels[mapped_task] = pred_label
                    except Exception as e:
                        print(f"处理任务 {task} 时出错: {e}")
        
        # 生成6位编码
        code = f"{max_labels['分叶']}{max_labels['棘刺']}{max_labels['纹理']}{max_labels['边缘']}{max_labels['钙化']}{max_labels['恶性程度']}"
        return code
    
    def get_physiotherapy_plan(self, classification_results: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        获取理疗方案
        - 参数:
            - classification_results: 分类结果列表
        - 返回:
            - Dict[str, str]: 理疗方案
        """
        code = self.generate_dialectical_code(classification_results)
        return self.dialectical_map.get(code, {
            '患者表征': '无对应辨证方案',
            '针灸治疗': 'none',
            '中药选择': '无',
            '食疗方案': '无'
        })

def dialectical_physiotherapy(classification_results: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    中医辨证理疗方案生成函数
    - 参数:
        - classification_results: 分类结果列表
    - 返回:
        - Dict[str, str]: 理疗方案
    """
    # 获取CSV文件路径
    csv_path = os.path.join(os.path.dirname(__file__), 'shu.csv')
    
    # 创建辨证实例
    dialectical = DialecticalPhysiotherapy(csv_path)
    
    # 获取理疗方案
    plan = dialectical.get_physiotherapy_plan(classification_results)
    
    return plan