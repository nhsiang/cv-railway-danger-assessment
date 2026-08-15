from ultralytics import YOLO

model = YOLO("runs/semantic/train/weights/best.pt")

def main():
    model.train(data="railway.yaml", epochs=35, imgsz=1024, device=0, cls_pw=0.6)

if __name__ == "__main__":
    main()