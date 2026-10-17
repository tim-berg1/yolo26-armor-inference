# yolo26-armor-inference
# yolo26-armor-inference
YOLO26n 装甲板关键点检测作业

## 作业说明
使用YOLO26n关键点模型 best.pt，识别装甲板B1/B2/B3/R1/R2/R3共6类目标，每个目标输出4个二维关键点。

1. 硬件环境：笔记本cpu
2. 模型文件：best.pt，从作业参考材料下载
3. 输入视频：input_video.mp4，本人独立拍摄装甲板实拍视频
4. 输出视频：output_result.mp4，模型推理可视化结果
5. 版本控制：1. **Python 版本**：Python 3.12
2. **主要依赖库**：
   - ultralytics >= 8.0：YOLO 模型推理框架
   - opencv-python：视频读写与可视化绘制
   - numpy：坐标与数值计算
   - torch：模型推理后端引擎

## 环境部署
    ```bash
- pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

## 拍摄说明
1. **拍摄地点**：西北农林科技大学水建学院 D414 实验室
2. **视频时长**：约9秒
3. **拍摄场景**：
   - 采用横屏 1080P 分辨率拍摄，画面主体为实验室放置的装甲板实物
   - 画面包含 B2、R3 两类指定装甲板目标
   - 拍摄过程中缓慢调整拍摄距离与俯仰角度，验证模型在不同远近、不同视角下的识别表现

## 实际识别表现、遇到的问题及处理方法
整体符合模型预期表现，未出现全片无有效检测的情况