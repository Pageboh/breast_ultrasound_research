## AutoDL 环境配置避坑指南
- **问题**：系统盘（30G）容易在安装 PyTorch 时报 Errno 28 空间不足。
- **解决方案**：
  1. 将 Conda 环境建在数据盘：`conda create --prefix /root/autodl-tmp/cv_seg_env python=3.9`
  2. 将 pip 临时解压目录改到数据盘：`export TMPDIR=/root/autodl-tmp/pip_tmp`

