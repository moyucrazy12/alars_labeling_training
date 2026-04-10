from ultralytics import YOLO

model = YOLO("runs_mixed_11n/M_stageA_1024/weights/best.pt")

results = model.train(
    data="data.yaml",
    task="obb",
    epochs=140,
    patience=35,

    imgsz=1280,     # good for 640x480 domain; less heavy than 960+
    rect=True,
    batch=6,

    degrees=2,
    translate=0.03,
    scale=0.08,
    shear=0.2,
    perspective=0.0,

    fliplr=0.4,
    flipud=0.0,

    hsv_h=0.005,
    hsv_s=0.10,
    hsv_v=0.10,

    mosaic=0.0,
    mixup=0.0,

    label_smoothing=0.0,

    optimizer="AdamW",
    lr0=0.00045,
    warmup_epochs=2,
    weight_decay=0.01,
    
    #cache = True,
    
    project="runs_mixed_11n",
    name="M_stageB_1280_finetune",
)

