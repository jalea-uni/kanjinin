import argparse
import cv2
import torch
import json
from torchvision import transforms
from model import get_model
from utils import segment_characters, render_template, compute_ssim


def load_model(path, num_classes, device):
    model = get_model(num_classes, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def infer_image(img_path, model, device, threshold, font_path, img_size):
    orig = cv2.imread(img_path)
    chars = segment_characters(orig)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    with open('label_map.json', 'r') as f:
        label_map = json.load(f)
    results = []
    for ch in chars:
        inp = transform(ch).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
            pred_idx = out.argmax(dim=1).item()
        # Map index -> code, then to int
        code_hex = label_map[str(pred_idx)]
        code_int = int(code_hex, 16)
        # Render perfect template and compute SSIM
        tpl = render_template(code_int, size=(img_size, img_size), font_path=font_path)
        ch_resized = cv2.resize(ch, (img_size, img_size))
        score = compute_ssim(ch_resized, tpl)
        quality = 'bene' if score >= threshold else 'male'
        results.append((chr(code_int), quality, score))
    return results


def main(args):
    # Load model
    with open('label_map.json', 'r') as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    model = load_model(args.model_path, num_classes, args.device)

    # Inference
    results = infer_image(
        args.image_path, model, args.device,
        args.threshold, args.font_path, args.img_size
    )
    for char, quality, score in results:
        print(f"Char: {char}, Quality: {quality}, SSIM: {score:.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='best_model.pth')
    parser.add_argument('--image-path', required=True)
    parser.add_argument('--threshold', type=float, default=0.75)
    parser.add_argument('--font-path', default=None)
    parser.add_argument('--img-size', type=int, default=64)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    main(args)