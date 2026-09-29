# SFA4D GPU 镜像：封装最佳模型（epoch 163 .pth）+ 训练/推理全流程
# 复刻实际环境：PyTorch 2.0.0+cu118 + CUDA 11.8（ONNX/INT8 产物不进镜像，走仓库分发的其他渠道）
# 构建：  docker build -t sfa4d .                      # 直连（apt/pip 已换清华源）
#         docker build --build-arg http_proxy=http://<主机IP>:7897 \
#                      --build-arg https_proxy=http://<主机IP>:7897 -t sfa4d .   # torch 走代理
# 运行：  docker run --gpus all -v /path/to/DRadDataset:/data/DRadDataset -it sfa4d <命令>
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 构建期网络加速（运行期在文件末尾清空）：apt/pip 走清华源，torch cu118 官方索引可走代理
ARG http_proxy=
ARG https_proxy=
ENV http_proxy=${http_proxy} https_proxy=${https_proxy}
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g; s@//.*security.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 与 requirements.txt 一致：CUDA 11.8 版 PyTorch 需从官方索引安装（清华 pypi 无 +cu118 本地版本）
# 注：本层会把 numpy 解析到 2.x，由下方 requirements 层的 numpy<2 约束统一降级（终态 1.26.x）
RUN pip3 install --no-cache-dir \
        torch==2.0.0+cu118 torchvision==0.15.1+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

# 其余依赖（跳过 torch/torchvision 两行，已单独安装；== 锚定避免误滤 torchsummary）
COPY requirements.txt .
RUN grep -vE '^(torch|torchvision)==' requirements.txt | pip3 install --no-cache-dir -r /dev/stdin

# 代码 + 最佳 .pth 权重（checkpoints/；onnx_models/ 与 quantized_models/ 被 .dockerignore 排除）
COPY . .
ENV MPLBACKEND=Agg

# 运行期清空构建期代理
ENV http_proxy= https_proxy=

# cudnn（torch 自带版）运行时按未版本化名 dlopen libnvrtc.so：
# runtime 镜像含该库（/usr/local/cuda-11.8/targets/.../libnvrtc.so.11.8.89）但缺符号链接且不在默认路径
RUN ln -sf /usr/local/cuda-11.8/targets/x86_64-linux/lib/libnvrtc.so.11.2 \
            /usr/local/cuda-11.8/targets/x86_64-linux/lib/libnvrtc.so
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib

# 默认：对挂载的数据集做超激进 NMS 推理（可用 docker run 覆盖任意命令）
CMD ["python3", "sfa/testing_export_ultra_aggressive.py", \
     "--pretrained_path", "./checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth", \
     "--dataset-dir", "/data/DRadDataset", \
     "--saved_fn", "sfa4d_163_ultra_aggressive", \
     "--peak_thresh", "0.25", "--nms_thresh", "0.2", "--gpu_idx", "0", \
     "--output-dir", "./results/sfa4d_163_ultra_aggressive"]
