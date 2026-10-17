from ultralytics import YOLO
import cv2

def main():
    # ========== 配置项，按需修改 ==========
    model_path = "best.pt"
    input_path = "input_video.mp4"
    output_path = "output_result.mp4"
    # =====================================

    # 加载YOLO26n关键点模型
    model = YOLO(model_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("视频打开失败，请检查路径！")
        return

    # 获取视频参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # 逐帧推理
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 推理，开启关键点
        results = model(frame, conf=0.3)

        for res in results:
            boxes = res.boxes
            keypoints = res.keypoints
            if keypoints is None:
                continue

            # 遍历每个装甲板目标
            for idx, box in enumerate(boxes):
                # 类别id + 类别名称
                cls_id = int(box.cls)
                cls_name = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 绘制检测框 + 类别文字
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, cls_name, (x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                # 获取4个关键点坐标
                kpts = keypoints.xy[idx].cpu().numpy()
                for (kx, ky) in kpts:
                    kx, ky = int(kx), int(ky)
                    cv2.circle(frame, (kx, ky), 4, (0,0,255), -1)

        out.write(frame)

    cap.release()
    out.release()
    print(f"推理完成！输出视频：{output_path}")

if __name__ == "__main__":
    main()

