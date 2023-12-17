from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.optim as optim
import argparse
from Unet import Load_unet, CustomDataset, WeightedBCEWithLogitsLoss

def main():
    # Command-line arguments
    parser = argparse.ArgumentParser(description='Train UNet with custom settings')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--res', type=str, default='full', help='Resolution in the format H,W')
    parser.add_argument('--epoch', type=int, default=10, help='number of training epochs')

    args = parser.parse_args()

    # Process resolution argument
    if args.res == 'full':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        resolution = (160, 240)
    else:
        try:
            res_value = int(args.res)
            resolution = (res_value, res_value)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(resolution, antialias=True),
                transforms.Resize((160, 240), antialias=True)
            ])
        except ValueError:
            raise ValueError("Invalid resolution value. Please provide 'full' or a single number.")

    if args.res == 'full':
        print("Training with Resolution: (160, 240)")
        filename_suffix = 'full'
    else:
        res_value = int(args.res)
        print(f"Training with Resolution: ({res_value}, {res_value})")
        filename_suffix = str(res_value)

    train_dataset = CustomDataset('unet_train/images', 'unet_train/masks', transform)

    unet_model = Load_unet()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_model.to(device)
    print(f"Training on device: {device}")

    weights = torch.exp(torch.linspace(0, 5, steps=49))

    criterion = WeightedBCEWithLogitsLoss(weights=weights)

    # you can finetune this part
    # unet_model.load_state_dict(torch.load('best_unet_model.pth'))

    optimizer = optim.Adam(unet_model.parameters(), lr=args.lr)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)

    num_epochs = args.epoch

    for epoch in range(num_epochs):
        unet_model.train()
        running_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:
            for images, masks in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")

                # Move data to GPU if available
                images, masks = images.to(device), masks.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = unet_model(images)
                loss = criterion(outputs, masks)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Update running loss
                running_loss += loss.item()

                tepoch.set_postfix(loss=loss.item())

        # Print average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    torch.save(unet_model.state_dict(), f'best_unet_model_{filename_suffix}.pth')

if __name__ == '__main__':
    main()