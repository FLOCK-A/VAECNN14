#!/usr/bin/env python3
"""
将旧格式的数据集JSON转换为新格式，添加必需的字段。

旧格式示例:
{
  "train": [{"file": "x.npy", "label": 0, "domain": 0}],
  "test": [{"file": "y.npy", "label": 1, "domain": 1}]
}

新格式示例:
{
  "train": [{
    "file": "x.npy",
    "scene_label": 0,
    "city_label": 0,
    "domain": 0,
    "split": "train",
    "device_id": "A"
  }],
  "test": [{
    "file": "y.npy",
    "scene_label": 1,
    "city_label": 1,
    "domain": 1,
    "split": "test",
    "device_id": "b"
  }]
}
"""
import json
import argparse
import os


# 设备ID映射 (可根据实际情况调整)
DEVICE_ID_MAP = {
    0: 'A',      # 源设备
    1: 'b',      # 目标设备1
    2: 'c',      # 目标设备2
    3: 's1',     # 目标设备3
    4: 's2',     # 目标设备4
    5: 's3',     # 目标设备5
    6: 's4',     # 未见设备1
    7: 's5',     # 未见设备2
    8: 's6',     # 未见设备3
}


def convert_sample(sample, split_name, default_city_label=None):
    """
    转换单个样本到新格式
    
    Args:
        sample: 原始样本字典
        split_name: 'train', 'val', 'test' 等
        default_city_label: 默认城市标签（如果没有指定）
        
    Returns:
        转换后的样本字典
    """
    # 如果已经是新格式，直接返回
    if 'scene_label' in sample and 'city_label' in sample:
        return sample
    
    new_sample = {}
    
    # 复制文件路径
    new_sample['file'] = sample['file']
    
    # 处理标签：假设旧的'label'是scene_label
    if 'label' in sample:
        new_sample['scene_label'] = sample['label']
        # 如果没有指定city_label，使用scene_label或默认值
        new_sample['city_label'] = sample.get('city_label', 
                                               default_city_label if default_city_label is not None 
                                               else sample['label'])
    else:
        new_sample['scene_label'] = sample.get('scene_label', -1)
        new_sample['city_label'] = sample.get('city_label', -1)
    
    # 域信息
    domain = sample.get('domain', 0)
    new_sample['domain'] = domain
    new_sample['device_id'] = sample.get('device_id', DEVICE_ID_MAP.get(domain, f'domain_{domain}'))
    
    # Split信息: test/val视为'test'，其他视为'train'
    if split_name.lower() in ['test', 'val', 'validation']:
        new_sample['split'] = 'test'
    else:
        new_sample['split'] = 'train'
    
    return new_sample


def convert_dataset_json(input_json, output_json=None, default_city_label=None):
    """
    转换整个数据集JSON文件
    
    Args:
        input_json: 输入JSON文件路径
        output_json: 输出JSON文件路径（None则覆盖输入文件）
        default_city_label: 默认城市标签
    """
    print(f"Loading dataset from: {input_json}")
    with open(input_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    converted = {}
    total_samples = 0
    
    for split_name, samples in dataset.items():
        print(f"Converting split '{split_name}': {len(samples)} samples")
        converted[split_name] = [
            convert_sample(s, split_name, default_city_label) 
            for s in samples
        ]
        total_samples += len(samples)
    
    # 输出文件路径
    if output_json is None:
        # 创建备份
        backup_path = input_json + '.backup'
        if not os.path.exists(backup_path):
            print(f"Creating backup: {backup_path}")
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
        output_json = input_json
    
    print(f"Saving converted dataset to: {output_json}")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    
    print(f"\nConversion complete!")
    print(f"Total samples converted: {total_samples}")
    print(f"Splits: {list(converted.keys())}")
    
    # 显示示例
    print("\n=== Sample from converted dataset ===")
    first_split = list(converted.keys())[0]
    if converted[first_split]:
        print(json.dumps(converted[first_split][0], indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        description='Convert old dataset JSON to new format with extended fields'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSON file path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path (default: overwrite input with backup)')
    parser.add_argument('--default_city_label', type=int, default=None,
                        help='Default city label if not present (default: use scene_label)')
    
    args = parser.parse_args()
    
    convert_dataset_json(args.input, args.output, args.default_city_label)


if __name__ == '__main__':
    main()
