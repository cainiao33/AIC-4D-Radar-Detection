import os
import sys

# sfa/ 内部模块以 `config.kitti_config`、`data_process.xxx` 等顶层名导入
# （与脚本内的 sys.path 自举约定一致），测试须把 sfa/ 加入 sys.path
SFA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sfa')
if SFA_DIR not in sys.path:
    sys.path.insert(0, SFA_DIR)
