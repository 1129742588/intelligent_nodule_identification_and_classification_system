import os
import numpy as np

from Lung_segmentation.model_operator import ModelOperator as lung_segmentation_operator
from nodules_identify.model_operate import ModelOperate as nodules_identify_operator
from nodule_classification.model_operate import NoduleModelService as nodule_classification_operator

from Lung_segmentation.data_processor import DataProcessor as lung_segmentation_data_processor
from nodules_identify.data_process import DataProcessor as nodules_identify_data_processor
from nodules_segmentation.predict import NoduleSegmentationPredictor

# 中医辨证模块
from Dialectical_physiotherapy.dialectical_physiotherapy import dialectical_physiotherapy as dialectical_physiotherapy_func



"""全局变量"""
# 使用脚本所在目录作为基础路径，确保路径解析可靠
temp_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_cache")
dataset_dir = r""
output_dir = None

lso = lung_segmentation_operator(dataset_dir)
lso.num_workers = 0
nio = nodules_identify_operator(dataset_dir, output_dir)
nco = nodule_classification_operator(dataset_dir, output_dir)

lso_data = lung_segmentation_data_processor(dataset_dir)
nio_data = nodules_identify_data_processor(dataset_dir, output_dir)
"""全局变量"""

"""-------------------------------------------------------------------"""


# 数据预加载
def load_data(ct_path):
    """
    加载CT图像数据，并返回图像数组和空间信息
    - 参数:
        - ct_path: CT图像文件路径，支持.mhd或.dicom格式，可以是dicom的文件夹路径
    - 返回:
        - ct_array: (N, H, W)的numpy数组，原始CT图像
        - ct_space_info: CT图像的空间信息字典，包含空间信息，用于后续重采样和坐标转换
        - ct_resampled: numpy数组，重采样后的CT图像
    """
    # 读取数据
    ct_array, ct_space_info = nio_data.load_data(ct_path)
    # 重采样
    ct_resampled = nio_data.resample_array(ct_array, ct_space_info)

    # 保存病例序号
    case_id = os.path.basename(ct_path)
    os.makedirs(temp_cache_dir, exist_ok=True)
    case_id_path = os.path.join(temp_cache_dir, "case_id.txt")
    with open(case_id_path, "w") as f:
        f.write(case_id)

    # 将原始CT图像、空间信息和重采样后的CT图像保存为npy文件
    save_dir = os.path.join(temp_cache_dir, "ct_data_array")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "ct_array.npy"), ct_array)
    np.save(os.path.join(save_dir, "ct_space_info.npy"), ct_space_info)
    np.save(os.path.join(save_dir, "ct_resampled.npy"), ct_resampled)

    return ct_array, ct_space_info, ct_resampled

# 肺部分割
def lung_segmentation(ct_array: np.ndarray, ct_space_info: dict):
    """
    分割肺部，并保存数据
    - 参数:
        - ct_array: (N, H, W)的numpy数组，原始CT图像
        - ct_space_info: CT图像的空间信息
    - 返回:
        - predicted_mask: (N, H, W)的numpy，二值化的肺部掩膜
    """
    # 判断输入类型

    # 推理肺部掩膜
    predicted_mask = lso.predict(ct_array)

    # 将数组和预测的掩膜保存为npy文件
    save_dir = os.path.join(temp_cache_dir, "lung_segmentation_cache")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "predicted_mask.npy"), predicted_mask)

    # 重采样并保存重采样后的数据
    mask_resampled = nio_data.resample_array(predicted_mask, ct_space_info)
    np.save(os.path.join(save_dir, "mask_resampled.npy"), mask_resampled)
    # print(f"重采样后的CT图像形状: {ct_resampled.shape}, 重采样后的掩膜形状: {mask_resampled.shape}")
    # print(f"占用内存：CT图像: {ct_resampled.nbytes / 1e6:.2f} MB, 掩膜: {mask_resampled.nbytes / 1e6:.2f} MB")

    return predicted_mask, mask_resampled

# 识别结节
def nodules_identify(ct_data, is_use_mask=True):
    """
    识别结节，并保存数据
    - 参数:
        - ct_data: CT图像数据，可以是文件路径或重采样后的numpy数组
        - is_use_mask: 布尔值，指示是否使用肺部掩膜
    - 返回:
        - identify_results: 包含各个结节字典的列表，identify_results[0]:
            -  key : value
            - "nodule_index": 结节索引，int类型，表示该样本在所有预测patch中的索引位置。
            - "nodule_prob": 结节概率，float类型，表示该样本被预测为结节的概率。
            - "patch_center": 区块中心坐标，格式为 (z, y, x)。
            - "patch_array": 区块数据，numpy数组，表示该样本所在区块的原始CT图像数据
    """
    # 结节信息保存路径
    save_dir = os.path.join(temp_cache_dir, "nodules_identify_cache")
    os.makedirs(save_dir, exist_ok=True)

    # 判断输入类型
    if isinstance(ct_data, str):
        # 重采样
        ct_array, sitk_information= nio_data.load_data(ct_data)
        ct_resampled = nio_data.resample_array(ct_array, sitk_information)
    elif isinstance(ct_data, np.ndarray):
        ct_resampled = ct_data
    else:
        raise ValueError("输入数据类型不支持，请提供文件路径或numpy数组")
    
    # 如果需要使用掩膜，加载重采样后的掩膜数据
    if is_use_mask:
        mask_path = os.path.join(temp_cache_dir, "lung_segmentation_cache", "mask_resampled.npy")
        if os.path.exists(mask_path):
            mask_resampled = np.load(mask_path)
        else:
            print("未找到重采样后的掩膜数据，继续使用原始CT数据进行识别")
            mask_resampled = None
    else:
        mask_resampled = None
    
    # 识别结节
    identify_results = nio.predict(ct_resampled, mask_resampled)
    # 将识别结果保存为npy文件
    np.save(os.path.join(save_dir, "identify_results.npy"), identify_results)

    # 结果显示
    print(f"识别到的结节信息: 共{len(identify_results)}个结节")
    for result in identify_results:
        print(f"结节索引 {result['nodule_index']}, 中心坐标 (z, y, x): {result['patch_center']}")

    return identify_results

# 分割结节
def nodules_segmentation(ct_data, identify_results: list):
    """
    分割结节，并保存数据
    - 参数:
        - ct_data: (N, H, W)的原始CT图像路径，或者重采样后的CT数组
        - identify_results: 包含结节信息的列表，来源于识别结节函数的输出，包含以下键：
            - patch_centers: 每个样本对应的区块中心坐标列表, 为(z, y, x)格式。
            - patches_array: 所有区块的数组，用于后续分析或可视化。
            - nodule_probs: 每个样本的结节概率数组。
            - nodule_nums: 预测为结节的样本数量。
    - 返回:
        - nodules_segmentation_results: 列表，每项是一个结节字典，包含：
            - patch_center: 结节中心坐标 (z, y, x)。
            - patch: 结节 patch。
            - patch_mask: mask。
            - full_mask: 全尺寸 mask。
    """
    # 获取结节中心坐标列表
    center_list = []
    if len(identify_results) > 0:
        for result in identify_results:
            center_list.append(result['patch_center'])
    else:
        print("未识别到结节，跳过分割步骤")
        return []

    # 实例化结节分割预测器
    predictor = NoduleSegmentationPredictor()
    nodules_segmentation_results = predictor.segment_from_coordinate(ct_input=ct_data, center_zyx=center_list)

    # 将分割结果保存为npy文件
    save_dir = os.path.join(temp_cache_dir, "nodules_segmentation_cache")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "nodules_segmentation_results.npy"), nodules_segmentation_results)

    return nodules_segmentation_results

# 分类结节
def nodule_classification(nodules_segmentation_results):
    """
    分类结节，并保存数据
    - 参数:
        - nodules_segmentation_results: 包含结节分割结果的列表，来源于分割结节函数的输出
    - 返回: 包含每个任务的预测类别和各类别概率的字典
        - 结构示例:
        - key: 任务名称
        - value: {
            "pred_label": 类别预测标签,
            "class_probabilities": 各标签概率字典
        }
        - class_probabilities 的 key 是类别标签（int），value 是对应的概率（float）。
        - 使用示例：
            - output["margin"]["pred_label"] 获取 margin 任务的预测类别标签
            - output["margin"]["class_probabilities"]["0"] 获取 margin 任务的0类别概率字典。

    """
    # 信息存储
    classification_results = []
    #数据构造
    for item in nodules_segmentation_results:
        # (D, H, W)的numpy数组
        patch = item['patch']
        patch_mask = item['patch_mask']
        # 仅保留结节区域
        nodule_96 = patch * patch_mask
        result = nco.predict_one(nodule_96)
        classification_results.append(result)
    
    # 将分类结果保存为npy文件
    save_dir = os.path.join(temp_cache_dir, "nodule_classification_cache")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "classification_results.npy"), classification_results)

    return classification_results

# 中医辨证
def dialectical_physiotherapy(classification_results):
    """
    中医辨证理疗方案生成
    - 参数:
        - classification_results: 分类结果列表
    - 返回:
        - Dict[str, str]: 理疗方案
    """
    # 调用中医辨证函数
    plan = dialectical_physiotherapy_func(classification_results)
    # 将辨证结果保存为JSON文件
    try:
        # 将辨证结果保存为JSON文件
        import json
        save_dir = os.path.join(temp_cache_dir, "physiotherapy_cache")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "dialectical_physiotherapy_results.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
    except Exception as e:
        pass
    
    return plan



# 综合判断

if __name__ =="__main__":

    """全局变量"""
    ct_path = r"D:\Learnfile\Dataset\LIDC-IDRI\LIDC-IDRI-0001\1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178\1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192"
    
    # file_name = r"1.3.6.1.4.1.14519.5.2.1.6279.6001.173106154739244262091404659845.mhd"
    # ct_path = os.path.join(r"D:\Learnfile\Dataset\LUNA16\subset1", file_name)




    """数据预先读取"""
    ct_array, ct_space_info, ct_resampled = load_data(ct_path)

    """1.肺部分割"""
    ct_array, predicted_mask = lung_segmentation(ct_array, ct_space_info)

    """2.识别结节"""
    identify_results = nodules_identify(ct_path, is_use_mask=True)

    """3.分割结节"""
    nodules_segmentation_results = nodules_segmentation(ct_path, identify_results)
        
    """4.分类结节"""
    classification_results = nodule_classification(nodules_segmentation_results)

    """5.中医辨证"""
    dialectical_plan = dialectical_physiotherapy(classification_results)

    """6.结果展示"""
    # 结节分类结果
    for idx, one_nodule_result in enumerate(classification_results):
        print("="*40,f"第 {idx+1} 个结节分类", "="*40)
        for task in one_nodule_result:
            pred_label = one_nodule_result[task]['pred_label']
            prob = one_nodule_result[task]['class_probabilities']
            print(f"任务: {task}, 预测标签: {pred_label} \n预测为正类的概率: {prob}")
            print("="*40)
    # 中医辨证理疗方案
    print("="*40,f"中医辨证理疗方案", "="*40)
    for key, value in dialectical_plan.items():
        print(f"{key}: {value}")
        print("="*40)