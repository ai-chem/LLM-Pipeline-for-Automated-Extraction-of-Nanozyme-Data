import glob
from pathlib import Path
from time import time

import click
import torch
from PIL import Image
from ultralytics import YOLO


def crop_and_save(image_path, boxes, save_dir, index):
    image = Image.open(image_path)
    for j, box in enumerate(boxes):
        xyxy = box.xyxy[0].cpu().numpy()
        cropped_image = image.crop((xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
        cropped_image.save(f"{save_dir}/{index}_{j}.png")
        print(f"Saved: {save_dir}/{index}_{j}.png")


@click.command()
@click.argument("yolo_weights", type=click.Path())
@click.argument("imgs_dir", type=click.Path())
@click.argument("results_dir", type=click.Path())
def main(yolo_weights: str, imgs_dir: str, results_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = YOLO(yolo_weights)
    model.to(device=device)

    png_list = glob.glob(glob.escape(imgs_dir) + "/*.png")
    test_list = [str(Path(path)) for path in png_list]

    start = time()
    print(test_list)
    results = model(test_list)
    end = time()
    print(f"mean time: {(end-start) / len(test_list)}")

    for i, res in enumerate(results):
        crop_and_save(test_list[i], res.boxes, results_dir, i)
        print(f"Classes: {res.boxes.cls}")
        print(f"XYXY Coords: {res.boxes.xyxy}")
        print()


if __name__ == "__main__":
    main()
