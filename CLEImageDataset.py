from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

class CLEImageDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def transformation(self, transforms):
        assert(self.transform is None)
        self.transform = transforms

    def __len__(self):
        return len(self.img_labels)

    def _get_image_path(self, idx):
        elements = self.img_labels.iloc[idx, 0].split()[0:2]
        disease, patient_id =elements[0], elements[-1][0:elements[1].index('-')]
        patient_dir = ' '.join([disease, patient_id])
        img_path = os.path.join(self.img_dir, patient_dir, self.img_labels.iloc[idx, 0])
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) # for non-distilledFrame folder
        return img_path
        
    def __getitem__(self, idx):
        # extract subdirectory
        image_name = self.img_labels.iloc[idx, 0]
        img_path = self._get_image_path(idx)

        # read the image from the location
        image = Image.open(img_path)

        # get the label
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label, image_name

if __name__ == '__main__':
#     # test if the custom dataset works
    all_data = CLEImageDataset('../cleanDistilledFrames', '../annotations.csv', transform=transforms.Compose([transforms.ToTensor()]))
    train_dataloader = DataLoader(all_data, batch_size=3, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")


# Iterate through dataloader to visualize
# Display image and label.
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")

# Training loop
