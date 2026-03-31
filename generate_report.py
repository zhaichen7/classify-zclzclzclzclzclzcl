import json
import os
from datetime import datetime

report = {
    '生成时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    '训练完成': True,
    '模型数量': len([d for d in os.listdir('.') if d.startswith('models_')]),
    '可用模型': {
        'RGB单模态': os.path.exists('./models_rgb_single/drought_best.pth'),
        'TIR单模态': os.path.exists('./models_tir_single/drought_best.pth'),
        'MS单模态': os.path.exists('./models_ms_single/drought_best.pth'),
        'RGB+TIR': os.path.exists('./models_rgb_tir/drought_best.pth'),
        'RGB+MS': os.path.exists('./models_rgb_ms/drought_best.pth'),
        'TIR+MS': os.path.exists('./models_tir_ms/drought_best.pth'),
    }
}

with open('training_report.json', 'w') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(json.dumps(report, indent=2, ensure_ascii=False))
print("\n✅ 报告已保存到 training_report.json")

