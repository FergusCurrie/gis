class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_list = [x for x in os.listdir(mask_dir)]
        self.image_list = ["18_"+x.replace('image', 'label').replace('npy', 'jpg') for x in os.listdir(mask_dir)]
        # print(self.image_list)
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])  # assuming masks have the same filename
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        mask = np.load(mask_path)


        if self.transform:
            augmented = self.transform(image=np.array(image), mask=np.array(mask))
            image, mask = augmented["image"], augmented["mask"]
        
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).float()
        
        return image, mask
