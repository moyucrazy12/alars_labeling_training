from ultralytics import YOLO

model = YOLO("yolo11n-obb.pt")
#model = YOLO("yolo_model.pt")

results = model.train(
    data="data.yaml",   # train = mixed; val you pick one domain or run 2 vals separately
    task="obb",
    epochs=260,
    patience=70,

    imgsz=1024,
    rect=False,
    batch=8,

    # geometry
    degrees=4,
    translate=0.05,
    scale=0.12,
    shear=0.5,
    perspective=0.0,

    fliplr=0.5,
    flipud=0.0,

    # photometric
    hsv_h=0.008,
    hsv_s=0.15,
    hsv_v=0.15,

    mosaic=0.15,
    mixup=0.0,
    close_mosaic=80,

    label_smoothing=0.01,

    optimizer="AdamW",
    lr0=0.0012,
    warmup_epochs=4,
    weight_decay=0.01,
    
    #cache = "disk",

    project="runs_mixed_11n",
    name="M_stageA_1024",
)

