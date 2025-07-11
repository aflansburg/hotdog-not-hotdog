from .utils import check_sys_arch


def predict(model_path: str, image_path: str):
    from ultralytics import YOLO
    import tempfile
    import requests

    device = check_sys_arch()

    if image_path.startswith(("http://", "https://")):
        print(f"Downloading image from URL: {image_path}")
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(image_path, headers=headers)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(response.content)
                temp_image_path = tmp_file.name

            image_path = temp_image_path
            print("Image downloaded successfully")
        except Exception as e:
            print(f"Error downloading image: {e}")
            return

    model = YOLO(model_path)

    results = model.predict(image_path, device=device, verbose=False)

    result = results[0]

    probs = result.probs

    predicted_class_id = probs.top1
    class_names = result.names
    predicted_class = class_names[predicted_class_id]
    confidence = probs.top1conf.item()

    if predicted_class == "hot_dog":
        print(f"🌭 HOT DOG! (confidence: {confidence:.1%})")
    else:
        print(f"❌ NOT HOT DOG (confidence: {confidence:.1%})")

    print(
        f"Probabilities: hot_dog: {probs.data[0]:.1%}, not_hot_dog: {probs.data[1]:.1%}"
    )
