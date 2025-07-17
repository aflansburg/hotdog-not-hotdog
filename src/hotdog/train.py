import os
from pathlib import Path
import shutil
from .utils import check_sys_arch


def validate(model_path: str) -> None:
    from ultralytics import YOLO

    device = check_sys_arch()
    data_path = "data/hotdog"

    print(f"Validating model: {model_path}")
    print(f"Using device: {device}")

    model = YOLO(model_path)
    results = model.val(data=data_path, device=device)

    print(f"Validation results: {results}")


def export_model(export_format: str = "pt", model_path: str = None) -> None:
    """
    Export an existing trained model to the models directory.

    Args:
        export_format: The format to export the model to.
        model_path: The path to the model to export.
    """
    from ultralytics import YOLO

    if model_path is None:
        model_path = "runs/classify/train/weights/best.pt"

    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Train a model first with: uv run hotdog --train")
        return

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    if export_format == "pt":
        final_path = models_dir / "hotdog_classifier.pt"
        shutil.copy(model_path, final_path)
        print(f"Model copied to {final_path}")
    else:  # torchscript
        model = YOLO(model_path)
        exported_model_path = model.export(format="torchscript")
        exported_path = Path(exported_model_path)
        final_path = models_dir / "hotdog_classifier.torchscript"
        shutil.move(exported_path, final_path)
        print(f"Model exported to {final_path}")


def train(
    export_format: str = "pt",
    model_size: str = "s",
    epochs: int = 200,
    byo_agent: bool = False,
) -> None:
    """
    Train a model and export it to the models directory.

    Args:
        export_format: The format to export the model to.
    """
    from ultralytics import YOLO, checks, hub

    if byo_agent:
        checks()
        hub.login(os.getenv("ULTRALYTICS_API_KEY"))
        model = YOLO(os.getenv("ULTRALYTICS_MODEL_URI"))
    else:
        model = YOLO(f"yolo11{model_size}-cls.pt")

    device = check_sys_arch()

    print(f"Training on device: {device}")

    data_path = "data/hotdog"

    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=640,
        device=device,
        augment=True,  # enables the following augmentations
        hsv_h=0.015,  # Hue aug
        hsv_s=0.7,  # Saturation aug
        hsv_v=0.4,  # Value aug
        degrees=20,  # More rotation (may? help w/ round foods? idk)
        translate=0.1,  # Translation aug
        scale=0.2,  # Scale aug
        fliplr=0.5,  # Horizontal flip probability
        dropout=0.2,  # Dropout for regularization
    )

    print(results)

    best_model_path = model.trainer.best
    print(f"Best model saved at: {best_model_path}")

    validate(best_model_path)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    if export_format == "pt":
        final_path = models_dir / "hotdog_classifier.pt"
        shutil.copy(best_model_path, final_path)
        print(f"Model copied to {final_path}")
    else:  # torchscript
        exported_model_path = model.export(format="torchscript")
        exported_path = Path(exported_model_path)
        final_path = models_dir / "hotdog_classifier.torchscript"
        shutil.move(exported_path, final_path)
        print(f"Model exported to {final_path}")
