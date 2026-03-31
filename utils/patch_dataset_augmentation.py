"""
给现有 dataset_drought.py 打补丁，添加高级数据增强
"""

# 读取原文件
with open('dataset_drought.py', 'r') as f:
    content = f.read()

# 在文件开头添加导入
if 'from advanced_augmentation import RemoteSensingAugmentation' not in content:
    import_pos = content.find('import torch')
    new_import = '\nimport sys\nsys.path.insert(0, "/home/zcl/addfuse")\nfrom advanced_augmentation import RemoteSensingAugmentation\n'
    content = content[:import_pos] + new_import + content[import_pos:]

# 修改 __init__ 方法，添加增强器
init_pattern = 'def __init__(self, csv_path, data_root, ids, augment=False,'
if 'self.augmentation = RemoteSensingAugmentation' not in content:
    # 找到 __init__ 中 self.augment = augment 的位置
    augment_pos = content.find('self.augment = augment')
    if augment_pos != -1:
        insert_pos = content.find('\n', augment_pos) + 1
        new_code = '''        # 高级数据增强
        if augment:
            self.augmentation = RemoteSensingAugmentation(p=0.7)
        else:
            self.augmentation = None
        
'''
        content = content[:insert_pos] + new_code + content[insert_pos:]

# 修改 __getitem__ 中的增强调用
if 'self.augmentation(rgb, tir, ms)' not in content:
    # 找到原有的 _augment 调用
    old_augment = 'if self.augment:\n            rgb, tir, ms = self._augment(rgb, tir, ms)'
    new_augment = '''if self.augment and self.augmentation is not None:
            rgb, tir, ms = self.augmentation(rgb, tir, ms)'''
    content = content.replace(old_augment, new_augment)

# 保存
with open('dataset_drought.py', 'w') as f:
    f.write(content)

print("✅ dataset_drought.py 已更新（添加高级数据增强）")
