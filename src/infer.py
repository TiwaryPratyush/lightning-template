import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
from src.models.dog_breed_classifier import DogBreedClassifier
from rich.progress import Progress
from src.utils.logging_utils import setup_logging

logger = setup_logging()

def load_model(ckpt_path: str):
    model = DogBreedClassifier.load_from_checkpoint(ckpt_path)
    model.eval()
    return model

def process_image(image_path: str, img_size: int = 224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def infer(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def main(args):
    logger.info("Starting inference")
    model = load_model(args.ckpt_path)
    class_names = model.hparams.class_names  # Assuming class names are stored in the model's hparams

    os.makedirs(args.output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(args.input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    with Progress() as progress:
        task = progress.add_task("[green]Processing images...", total=len(image_files))

        for filename in image_files:
            image_path = os.path.join(args.input_folder, filename)
            image_tensor = process_image(image_path)
            predicted_class_idx = infer(model, image_tensor)
            predicted_class = class_names[predicted_class_idx]

            output_path = os.path.join(args.output_folder, f"{os.path.splitext(filename)[0]}_{predicted_class}.txt")
            with open(output_path, 'w') as f:
                f.write(f"Predicted class: {predicted_class}")

            logger.info(f"Processed {filename}: Predicted class - {predicted_class}")
            progress.update(task, advance=1)

    logger.info("Inference completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dog Breed Classification Inference")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing input images")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the folder to save prediction results")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the model checkpoint")
    args = parser.parse_args()
    main(args)
