from transformers import Dinov2Model, Dinov2PreTrainedModel
import os
from sklearn.model_selection import train_test_split
import albumentations as A
from torch.utils.data import DataLoader
from gis.config import Config
from PIL import Image
import torch
from torch.utils.data import Dataset
import segmentation_models_pytorch as smp
import numpy as np

config = Config()

MODEL_CONFIG = {
	"batch_size": 4,
	"epochs": 5,
	"learning_rate": 1e-4,
	"val_split": 0.2,
	"num_workers": 4,
}
id2label = {
	0: "baclgrpimd",
	1: "track",
}


TOKEN_WIDTH = 18  #


def get_image_and_mask_files():
	mask_files = os.listdir(config.mnt_path / "label/18")
	coords = []
	for mask in mask_files:
		x, y = mask.split("_")
		x, y = int(x), int(y.replace(".npy", ""))
		coords.append((x, y))
	image_files = [f"18_{x}_{y}.jpg" for (x, y) in coords]
	return image_files, mask_files


class Dinov2FeatureExtractor(Dinov2PreTrainedModel):
	def __init__(self, config):
		super().__init__(config)

		self.dinov2 = Dinov2Model(config)
		# self.classifier = LinearClassifier(config.hidden_size, 32, 32, config.num_labels)

	def forward(
		self,
		pixel_values,
		output_hidden_states=False,
		output_attentions=False,
		labels=None,
	):
		# use frozen features
		outputs = self.dinov2(
			pixel_values,
			output_hidden_states=output_hidden_states,
			output_attentions=output_attentions,
		)
		# get the patch embeddings - so we exclude the CLS token
		patch_embeddings = outputs.last_hidden_state[:, 1:, :]
		return patch_embeddings


class SegmentationDataset(Dataset):
	def __init__(self, images, masks, transform=None):
		self.images = images
		self.masks = masks
		self.transform = transform
		self.image_dir = config.mnt_path / "image/18"
		self.mask_dir = config.mnt_path / "label/18"

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		img_path = os.path.join(self.image_dir, self.images[idx])
		mask_path = os.path.join(self.mask_dir, self.masks[idx])
		image = Image.open(img_path).convert("RGB")
		original_image = np.array(image)
		original_mask = np.load(mask_path)

		transformed = self.transform(image=original_image, mask=original_mask)
		image, target = image, target = (
			torch.tensor(transformed["image"]),
			torch.LongTensor(transformed["mask"]),
		)
		image = image.permute(2, 0, 1)
		return image, target, original_image, original_mask


def get_loaders():
	image_files, mask_files = get_image_and_mask_files()

	train_images, val_images, train_masks, val_masks = train_test_split(
		image_files, mask_files, test_size=MODEL_CONFIG["val_split"], random_state=42
	)

	ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
	ADE_STD = np.array([58.395, 57.120, 57.375]) / 255
	width = 256

	train_transform = A.Compose(
		[
			A.Resize(width=width, height=width),
			A.HorizontalFlip(p=0.5),
			A.Normalize(mean=ADE_MEAN, std=ADE_STD),
		]
	)

	val_transform = A.Compose(
		[
			A.Resize(width=width, height=width),
			A.Normalize(mean=ADE_MEAN, std=ADE_STD),
		]
	)
	train_dataset = SegmentationDataset(
		train_images, train_masks, transform=train_transform
	)
	val_dataset = SegmentationDataset(val_images, val_masks, transform=val_transform)

	train_loader = DataLoader(train_dataset, batch_size=MODEL_CONFIG["batch_size"])
	val_loader = DataLoader(val_dataset, batch_size=MODEL_CONFIG["batch_size"])
	return train_loader, val_loader


class LinearClassifier(torch.nn.Module):
	def __init__(
		self, in_channels, tokenW=TOKEN_WIDTH, tokenH=TOKEN_WIDTH, num_labels=1
	):
		super(LinearClassifier, self).__init__()

		self.in_channels = in_channels
		self.width = tokenW
		self.height = tokenH
		self.mixer = torch.nn.Conv2d(in_channels, 128, (3, 3))
		self.classifier = torch.nn.Conv2d(128, num_labels, (1, 1))

	def forward(self, embeddings):
		embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
		embeddings = embeddings.permute(0, 3, 1, 2)

		return self.classifier(self.mixer(embeddings))


class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
	def __init__(self, config):
		super().__init__(config)

		self.dinov2 = Dinov2Model(config)
		self.classifier = LinearClassifier(
			config.hidden_size, TOKEN_WIDTH, TOKEN_WIDTH, 1
		)

	def forward(
		self,
		pixel_values,
		output_hidden_states=False,
		output_attentions=False,
		labels=None,
	):
		# use frozen features
		outputs = self.dinov2(
			pixel_values,
			output_hidden_states=output_hidden_states,
			output_attentions=output_attentions,
		)
		# get the patch embeddings - so we exclude the CLS token
		# cls_embeddings = outputs.last_hidden_state[:, 0, :]
		patch_embeddings = outputs.last_hidden_state[:, 1:, :]
		logits = self.classifier(patch_embeddings)
		logits = torch.nn.functional.interpolate(
			logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False
		)
		return logits


def calculate_metrics(pred, target, threshold=0.5):
	"""Calculate IoU, Dice, and other metrics."""
	pred_binary = (pred > threshold).float()
	target_binary = target.float()

	# IoU
	intersection = (pred_binary * target_binary).sum()
	union = pred_binary.sum() + target_binary.sum() - intersection
	iou = intersection / (union + 1e-8)

	# Dice coefficient
	dice = (2 * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-8)

	# Pixel accuracy
	correct = (pred_binary == target_binary).sum()
	total = target_binary.numel()
	accuracy = correct / total

	return {"iou": iou.item(), "dice": dice.item(), "accuracy": accuracy.item()}


def train_epoch(model, train_loader, criterion, optimizer, device):
	"""Train for one epoch."""
	model.train()
	train_loss = 0.0
	train_metrics = {"iou": 0.0, "dice": 0.0, "accuracy": 0.0}

	for images, masks, _, _ in train_loader:
		images = images.to(device)
		masks = masks.to(device)

		optimizer.zero_grad()

		outputs = model(images)
		# print(outputs.shape, masks.shape, masks.unsqueeze(1).shape)
		loss = criterion(outputs, masks.unsqueeze(1))
		loss.backward()
		optimizer.step()

		train_loss += loss.item()

		# Calculate metrics
		with torch.no_grad():
			batch_metrics = calculate_metrics(
				torch.sigmoid(outputs), masks.unsqueeze(1)
			)
			for key in train_metrics:
				train_metrics[key] += batch_metrics[key]

	train_loss /= len(train_loader)
	for key in train_metrics:
		train_metrics[key] /= len(train_loader)

	return train_loss, train_metrics


def validate_model(model, val_loader, criterion, device):
	"""Validate the model."""
	model.eval()
	val_loss = 0.0
	val_metrics = {"iou": 0.0, "dice": 0.0, "accuracy": 0.0}

	with torch.no_grad():
		for images, masks, _, _ in val_loader:
			images = images.to(device)
			masks = masks.to(device)

			outputs = model(images)
			loss = criterion(outputs, masks.unsqueeze(1))  # Add channel dim for masks

			val_loss += loss.item()

			# Calculate metrics
			batch_metrics = calculate_metrics(outputs, masks.unsqueeze(1))
			for key in val_metrics:
				val_metrics[key] += batch_metrics[key]

	# Average metrics
	val_loss /= len(val_loader)
	for key in val_metrics:
		val_metrics[key] /= len(val_loader)

	return val_loss, val_metrics


def train_loop():
	model = Dinov2ForSemanticSegmentation.from_pretrained(
		"facebook/dinov2-base", id2label=id2label, num_labels=len(id2label)
	)
	for name, param in model.named_parameters():
		if name.startswith("dinov2"):
			param.requires_grad = False

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	optimizer = torch.optim.AdamW(model.parameters(), lr=MODEL_CONFIG['learning_rate'])
	loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

	model = model.to(device)


	train_loader, val_loader = get_loaders()


	history = {
		'train_loss': [], 'val_loss': [],
		'train_iou': [], 'val_iou': [],
		'train_dice': [], 'val_dice': [],
		'train_accuracy': [], 'val_accuracy': []
	}

	for epoch in range(50):
		train_loss, train_metrics = train_epoch(
			model, train_loader, loss_fn, optimizer, device
		)
		val_loss, val_metrics = validate_model(model, val_loader, loss_fn, device) # todo: get eval loaders 

		history['train_loss'].append(train_loss)
		history['val_loss'].append(val_loss)
		history['train_iou'].append(train_metrics['iou'])
		history['val_iou'].append(val_metrics['iou'])
		history['train_dice'].append(train_metrics['dice'])
		history['val_dice'].append(val_metrics['dice'])
		history['train_accuracy'].append(train_metrics['accuracy'])
		history['val_accuracy'].append(val_metrics['accuracy'])

		print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_metrics['iou']:.4f}")
		print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_metrics['iou']:.4f}")





# def pca_feature_map():
# 	import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import cv2
# from pathlib import Path


# for i in range(8):
#     features = outputs[i].numpy()
#     feature_dim = 768

#     patches = features.reshape(-1, feature_dim)
#     print(patches.shape)

#     scaler = StandardScaler()
#     patches_scaled = scaler.fit_transform(patches)

#     # Apply PCA to reduce to 3 components (RGB)
#     pca = PCA(n_components=3, random_state=42)
#     pca_result = pca.fit_transform(patches_scaled) 
#     pca_result.shape

#     pca_min = pca_result.min(axis=0)
#     pca_max = pca_result.max(axis=0)
#     pca_normalized = (pca_result - pca_min) / (pca_max - pca_min + 1e-8)


#     pca_spatial = pca_normalized.reshape(18, 18, 3)
#     pca_spatial.shape

#     fig, axs = plt.subplots(1,3,figsize=(15,5))
#     axs[0].imshow(batch_image[i].permute(1,2,0))
#     axs[1].imshow(pca_spatial)
#     axs[2].imshow(batch_original_image[i])
#     plt.show()