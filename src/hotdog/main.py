import argparse
import os
from .train import train, validate, export_model
from .predict import predict

DEV = os.getenv("DEV", "False").lower() == "true"

if DEV:
    print("Running in dev mode.\n Loading .env.local")
    from dotenv import load_dotenv

    load_dotenv(".env.local")

ULTRALYTICS_BYO_AGENT = False

if os.getenv("ULTRALYTICS_API_KEY"):
    print("ULTRALYTICS_API_KEY is set - using 'bring your own agent mode'")
    ULTRALYTICS_BYO_AGENT = True
else:
    print("ULTRALYTICS_API_KEY is not set and will not communicate with Ultralytics")


def main():
    parser = argparse.ArgumentParser(description="Ultralytics YOLO model runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train the model")
    group.add_argument(
        "--predict", type=str, metavar="IMAGE_PATH", help="Run prediction on image path"
    )
    group.add_argument(
        "--validate", type=str, metavar="MODEL_PATH", help="Validate a trained model"
    )
    group.add_argument(
        "--export", action="store_true", help="Export trained model to models directory"
    )

    parser.add_argument(
        "--model-size",
        type=str,
        choices=["n", "s", "m", "l", "x"],
        default="s",
        help="Model size - eg. (n=YOLO11n, s=YOLO11s, m=YOLO11m, l=YOLO11l, x=YOLO11x)",
    )

    parser.add_argument(
        "--model", type=str, help="Path to model file (e.g., yolo11n-cls.pt)"
    )
    parser.add_argument(
        "--export-format",
        type=str,
        choices=["pt", "torchscript"],
        default="pt",
        help="Export format for trained model (default: pt)",
    )

    args = parser.parse_args()

    if args.train:
        if args.model:
            parser.error("--model is not used with --train")

        epochs = input("Enter the number of epochs to train for: [200] ") or 200

        print(f"Training mode selected with model size: {args.model_size}")
        train(
            export_format=args.export_format,
            model_size=args.model_size,
            epochs=epochs,
            byo_agent=ULTRALYTICS_BYO_AGENT,
        )
        return
    elif args.predict:
        if not args.model:
            parser.error("--model is required when using --predict")
        print(f"Prediction mode selected for image: {args.predict}")
        print(f"Using model: {args.model}")

        predict(args.model, args.predict)
    elif args.validate:
        if args.model:
            parser.error(
                "--model is not used with --validate (model path is provided as argument)"
            )
        print("Validation mode selected")
        validate(args.validate)
    elif args.export:
        if args.model:
            parser.error("--model is not used with --export")
        print("Export mode selected")
        export_model(export_format=args.export_format)


if __name__ == "__main__":
    main()
