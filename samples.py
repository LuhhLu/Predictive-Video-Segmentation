from torchvision import transforms
from Unet import CustomDataset
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def get_bounding_box(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def main():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = CustomDataset('unet_train/images', 'unet_train/masks', transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    mask_count = {i: 0 for i in range(1, 49)}
    output_dir = "samples"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range(1, 49):
        os.makedirs(os.path.join(output_dir, f"label_{i}"), exist_ok=True)

    print('generating samples')

    for image, mask in tqdm(train_loader, desc="Collecting Masks"):
        # print("Image shape:", image[0].shape)  # Debug: Print image shape

        mask_array = mask.numpy()[0]
        for mask_type in range(1, 49):
            mask_type_array = mask_array[mask_type]
            if np.any(mask_type_array == 1) and mask_count[mask_type] < 5:
                rmin, rmax, cmin, cmax = get_bounding_box(mask_type_array)

                # Debug: Print cropping coordinates
                # print("Cropping coords:", rmin, rmax, cmin, cmax)

                # Ensure cropping coords are within bounds
                height, width = image[0].shape[1], image[0].shape[2]
                rmin, rmax = max(0, rmin), min(height, rmax)
                cmin, cmax = max(0, cmin), min(width, cmax)

                # Ensure rmin < rmax and cmin < cmax
                if rmin >= rmax or cmin >= cmax:
                    continue  # Skip if cropping coords are invalid

                cropped_image = TF.to_pil_image(image[0][:, rmin:rmax, cmin:cmax])
                cropped_image.save(os.path.join(output_dir, f"label_{mask_type}", f"{mask_count[mask_type]}.png"))
                mask_count[mask_type] += 1
                break

        if all(count >= 5 for count in mask_count.values()):
            break

    print('generating distributions')

    for image, mask in tqdm(train_loader, desc="Collecting Masks"):
        mask_array = mask.numpy()[0]
        for mask_type in range(1, 49):
            mask_type_array = mask_array[mask_type]
            if np.any(mask_type_array == 1):
                mask_count[mask_type] += 1
                break

    mask_types = list(mask_count.keys())
    counts = list(mask_count.values())

    plt.figure(figsize=(12, 6))
    plt.bar(mask_types, counts, color='skyblue')
    plt.xlabel('Mask Type')
    plt.ylabel('Count')
    plt.title('Distribution of Mask Types')
    plt.xticks(mask_types)

    # Saving the figure to a file
    plt.savefig('distribution.png')

if __name__ == '__main__':
    main()