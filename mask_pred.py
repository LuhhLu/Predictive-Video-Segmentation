import torch
import numpy as np
from Unet import Load_unet
from tqdm import tqdm
import os
from torchvision import transforms
from PIL import Image
import torchmetrics
import cv2
import argparse

jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)


def compute_centroids(mask, object_values):
    centroids = {}
    for value in object_values:
        object_mask = (mask == value)
        M = cv2.moments(object_mask.astype(np.uint8))
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids[value] = np.array([cX, cY])
    return centroids

frame_size = (240, 160)  # 假设帧的尺寸是240x160

def compute_distance(centroid1, centroid2):
    return np.sqrt(np.sum((centroid1 - centroid2) ** 2))


def find_closest_centroid(centroid, centroids, threshold):
    dists = {v: compute_distance(centroid, c) for v, c in centroids.items()}
    mindist = min(dists.values())
    if mindist <= threshold:
        return min(dists, key=dists.get)
    else:
        return 0


def is_near_edge(centroid, edge_margin):
    x_near_edge = centroid[0] < edge_margin or frame_size[0] - centroid[0] < edge_margin
    y_near_edge = centroid[1] < edge_margin or frame_size[1] - centroid[1] < edge_margin
    return x_near_edge or y_near_edge


def process_disappeared_elements(frame_idx, prev, centroids, frames, config):
    edge_margin = config.edge_margin
    threshold = config.threshold
    disappeared_elements = set(prev.keys()) - set(centroids.keys())
    #     print('disappeared',disappeared_elements)
    for value_p in disappeared_elements:
        centroid_p = prev[value_p]
        if not is_near_edge(centroid_p,edge_margin):
            min_value_key = find_closest_centroid(centroid_p, centroids,threshold)
            if min_value_key != 0:
                count_1 = np.count_nonzero(frames[frame_idx] == min_value_key)
                count_2 = np.count_nonzero(frames[frame_idx - 1] == value_p)
                if count_2 * 0.8 < count_1 < count_2 * 1.2:
                    frames[frame_idx][frames[frame_idx] == min_value_key] = value_p
                else:
                    frames[frame_idx][frames[frame_idx - 1] == value_p] = value_p
            else:
                frames[frame_idx][frames[frame_idx - 1] == value_p] = value_p

    unique_numbers = np.unique(frames[frame_idx])
    count_dict = {num: np.count_nonzero(frames[frame_idx] == num) for num in unique_numbers}
    anomalies = [key for key, value in count_dict.items() if value < 50]
    for anomaly in anomalies:
        frames[frame_idx][frames[frame_idx] == anomaly] = 0


def process_new_elements(frame_idx, prev, centroids, frames, config):
    threshold=config.threshold
    edge_margin = config.edge_margin
    new_elements = set(centroids.keys()) - set(prev.keys())
    #     print('new',new_elements)
    for value in new_elements:
        centroid = centroids[value]
        if frame_idx < 11 and is_near_edge(centroid,edge_margin):
            continue
        else:
            min_value_key = find_closest_centroid(centroid, prev,threshold)
            if min_value_key != 0:
                frames[frame_idx][frames[frame_idx] == value] = min_value_key


def consistency(frames,config):
    unique_objects = np.unique(frames)
    unique_objects = unique_objects[unique_objects != 0]
    prev = {}

    for idx in range(frames.shape[0]):
        centroids = compute_centroids(frames[idx], unique_objects)

        if idx == 0:
            prev = centroids
        else:
            process_disappeared_elements(idx, prev, centroids, frames,config)
            centroids = compute_centroids(frames[idx], unique_objects)
            process_new_elements(idx, prev, centroids, frames,config)
            prev = compute_centroids(frames[idx], unique_objects)

    return frames

def Anomalies_drop(masks):
    for idx in range(len(masks)):
        unique_numbers = np.unique(masks[idx])
        count_dict = {num: np.count_nonzero(masks[idx] == num) for num in unique_numbers}
        anomalies = [key for key, value in count_dict.items() if value < 50]
        for anomaly in anomalies:
            masks[idx][masks[idx] == anomaly] = 0
    return masks


def main():
    # Command-line arguments
    parser = argparse.ArgumentParser(description='Train UNet with custom settings')
    parser.add_argument('--val', action='store_true', help='Enable validation (default: enabled)')
    parser.add_argument('--no-val', action='store_false', dest='val', help='Disable validation')
    parser.set_defaults(val=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Predicting on {device}')
    unet_model = Load_unet('best_unet_model_full.pth').to(device)
    unet_model_64 = Load_unet('best_unet_model_64.pth').to(device)

    if args.val:
        pred = np.load('pred_val.npy')
        val_y = []
        with open('val_paths.txt', 'r') as file:
            for line in file:
                val_y.append(line.strip())
    else:
        pred = np.load('pred_hidden.npy')
        val_y = []
        with open('hidden_paths.txt', 'r') as file:
            for line in file:
                val_y.append(line.strip())

    video_indices = {int(path.split('_')[-1]): idx for idx, path in enumerate(val_y)}
    sorted_indices = sorted(video_indices.keys())
    val_y = [val_y[video_indices[idx]] for idx in sorted_indices]
    pred = np.array([pred[video_indices[idx]] for idx in sorted_indices])

    # predicting prev11 frames

    print('Predicting prev11 frames!')

    unet_model.eval()

    with torch.no_grad():

        for idx in tqdm(range(len(val_y)), desc="Processing videos"):

            video_path = str(val_y[idx])

            images = [f for f in os.listdir(video_path) if f.endswith('.png')]
            images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            transform = transforms.ToTensor()
            frames = []
            for image_file in images[0:11]:
                image_path = os.path.join(video_path, image_file)
                image = Image.open(image_path).convert("RGB")
                image_tensor = transform(image)
                frames.append(image_tensor)
            frames = torch.stack(frames, dim=0)
            frames = frames.to(device)
            pred_masks = unet_model(frames)
            pred_masks = torch.sigmoid(pred_masks)
            pred_masks = pred_masks > 0.5
            pred_masks_np = pred_masks.cpu().numpy()
            pred_masks_np = np.argmax(pred_masks_np, axis=1).astype(np.uint8)
            pred_masks_np = Anomalies_drop(pred_masks_np)
            save_path = os.path.join(video_path, 'mask_pred.npy')
            np.save(save_path, pred_masks_np.astype(np.uint8))

    unet_model_64.eval()

    with torch.no_grad():
        frame_22_pred_unconsis = []
        frame_22_pred = []
        if args.val:
            frame_22_gt = []

        for idx in tqdm(range(len(val_y)), desc="Processing videos"):
            video_path = str(val_y[idx])
            images = pred[idx]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((160, 240), antialias=True)
            ])
            frames = []
            for image in images[0:11]:
                image = np.transpose(image, (1, 2, 0))
                image_tensor = transform(image)
                frames.append(image_tensor)
            frames = torch.stack(frames, dim=0)
            frames = frames.to(device)
            pred_masks = unet_model_64(frames)
            pred_masks = torch.sigmoid(pred_masks)
            pred_masks = pred_masks > 0.5
            pred_masks_np = pred_masks.cpu().numpy()
            pred_masks_np = np.argmax(pred_masks_np, axis=1).astype(np.uint8)
            pred_masks_np = Anomalies_drop(pred_masks_np)
            frame_22_pred_unconsis.append(pred_masks_np[-1])
            if args.val:
                mask_path = os.path.join(video_path, 'mask.npy')
                mask = np.load(mask_path)
                frame_22_gt.append(mask[-1])
            save_path = os.path.join(video_path, 'mask_pred_2.npy')
            np.save(save_path, pred_masks_np.astype(np.uint8))
            frame_prev_path = os.path.join(video_path, 'mask_pred.npy')
            frame_prev = np.load(frame_prev_path)
            new_frames = np.concatenate([frame_prev, pred_masks_np.astype(np.uint8)], axis=0)
            frames_consis = consistency(new_frames,args)
            save_path = os.path.join(video_path, 'final_mask_pred.npy')
            np.save(save_path, frames_consis)
            frame_22_pred.append(frames_consis[-1])
        frame_22_pred_unconsis = np.stack(frame_22_pred_unconsis, axis=0)
        frame_22_pred = np.stack(frame_22_pred, axis=0)
        if args.val:
            frame_22_gt = np.stack(frame_22_gt, axis=0)
            np.save('val_22_gt.npy', frame_22_gt.astype(np.uint8))
            print('the final result:', jaccard(torch.tensor(frame_22_pred), torch.tensor(frame_22_gt)))
            np.save('val_22_pred_unconsis.npy', frame_22_pred_unconsis.astype(np.uint8))
            np.save('val_22_pred.npy', frame_22_pred.astype(np.uint8))
        else:
            np.save('hidden_22_pred_unconsis.npy', frame_22_pred_unconsis.astype(np.uint8))
            np.save('hidden_22_pred.npy', frame_22_pred.astype(np.uint8))

if __name__ == '__main__':
    main()