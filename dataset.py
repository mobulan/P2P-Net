import imageio as reader
import os
import scipy.io as sio
from PIL import Image
from torchvision import transforms


import scipy
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import *


# from config import INPUT_SIZE
INPUT_SIZE = (448, 448)


class CUB():
	def __init__(self, root, is_train=True, data_len=None):
		self.root = root
		self.is_train = is_train
		self.preload = False

		img_txt_file = open(os.path.join(self.root, 'images.txt'))
		label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
		train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))

		img_name_list = []
		for line in img_txt_file:
			img_name_list.append(line[:-1].split(' ')[-1])

		label_list = []
		for line in label_txt_file:
			label_list.append(int(line[:-1].split(' ')[-1]) - 1)

		train_test_list = []
		for line in train_val_file:
			train_test_list.append(int(line[:-1].split(' ')[-1]))

		self.train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
		self.test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]

		self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
		self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

		if self.preload:
			if self.is_train:
				self.train_img = [reader.imread(os.path.join(self.root, 'images', train_file)) for train_file in
				                  self.train_file_list[:data_len]]
				self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
			else:
				self.test_img = [reader.imread(os.path.join(self.root, 'images', test_file)) for test_file in
				                 self.test_file_list[:data_len]]
				self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

	def __getitem__(self, index):
		if self.is_train:
			if self.preload:
				img, target = self.train_img[index], self.train_label[index]
			else:
				img_path = os.path.join(self.root, 'images', self.train_file_list[index])
				img, target = reader.imread(img_path), self.train_label[index]
			if len(img.shape) == 2:
				img = np.stack([img] * 3, 2)
			img = Image.fromarray(img, mode='RGB')
			# img = transforms.Resize((600, 600), Image.BILINEAR)(img)
			# img = transforms.RandomCrop(INPUT_SIZE)(img)
			# img = transforms.RandomHorizontalFlip()(img)
			img = transforms.Resize((550, 550))(img)
			img = transforms.RandomCrop(INPUT_SIZE, padding=8)(img)
			img = transforms.RandomHorizontalFlip()(img)
			img = transforms.ToTensor()(img)
			# img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
			img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)

		else:
			if self.preload:
				img, target = self.test_img[index], self.test_label[index]
			else:
				img_path = os.path.join(self.root, 'images', self.test_file_list[index])
				img, target = reader.imread(img_path), self.test_label[index]
			if len(img.shape) == 2:
				img = np.stack([img] * 3, 2)
			img = Image.fromarray(img, mode='RGB')
			# img = transforms.Resize((600, 600), Image.BILINEAR)(img)
			# img = transforms.CenterCrop(INPUT_SIZE)(img)
			# img = transforms.ToTensor()(img)
			# img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
			img = transforms.Resize((550, 550))(img)
			img = transforms.CenterCrop(INPUT_SIZE)(img)
			img = transforms.ToTensor()(img)
			img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)

		return img, target

	def __len__(self):
		if self.is_train:
			return len(self.train_label)
		else:
			return len(self.test_label)


class CAR():
	def __init__(self, root, is_train=True, data_len=None):
		self.root = root
		self.is_train = is_train

		data = sio.loadmat(os.path.join(root, 'Stanford_cars_annos.mat'))
		annotations = data['annotations'].squeeze(
			0)  # ('relative_im_path, 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'test')
		class_names = data['class_names']

		self.train_file_list = [ann[0][0] for ann in annotations if ann[-1] == 0]
		self.train_label = [ann[-2][0][0].astype('int64') - 1 for ann in annotations if ann[-1] == 0]
		self.train_bbox = [(ann[1][0][0], ann[2][0][0], ann[3][0][0], ann[4][0][0]) for ann in annotations if
		                   ann[-1] == 0]

		self.test_file_list = [ann[0][0] for ann in annotations if ann[-1] == 1]
		self.test_label = [ann[-2][0][0].astype('int64') - 1 for ann in annotations if ann[-1] == 1]
		self.test_bbox = [(ann[1][0][0], ann[2][0][0], ann[3][0][0], ann[4][0][0]) for ann in annotations if
		                  ann[-1] == 1]

	def __getitem__(self, index):
		if self.is_train:
			img_path = os.path.join(self.root, self.train_file_list[index])
			bbox = self.train_bbox[index]
			x1, y1, x2, y2 = bbox
			img, target = reader.imread(img_path), self.train_label[index]
			if len(img.shape) == 2:
				img = np.stack([img] * 3, 2)
				img = img[y1:y2, x1:x2, :]
			img = Image.fromarray(img, mode='RGB')
			# img = transforms.Resize((600, 600), Image.BILINEAR)(img)
			# img = transforms.RandomCrop(INPUT_SIZE)(img)
			# img = transforms.RandomHorizontalFlip()(img)
			img = transforms.Resize((550, 550))(img)
			img = transforms.RandomCrop(INPUT_SIZE, padding=8)(img)
			img = transforms.RandomHorizontalFlip()(img)
			img = transforms.ToTensor()(img)
			# img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
			img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)

		else:
			img_path = os.path.join(self.root, self.test_file_list[index])
			bbox = self.test_bbox[index]
			x1, y1, x2, y2 = bbox
			img, target = reader.imread(img_path), self.test_label[index]
			if len(img.shape) == 2:
				img = np.stack([img] * 3, 2)
				img = img[y1:y2, x1:x2, :]
			img = Image.fromarray(img, mode='RGB')
			# img = transforms.Resize((600, 600), Image.BILINEAR)(img)
			# img = transforms.CenterCrop(INPUT_SIZE)(img)
			# img = transforms.ToTensor()(img)
			# img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
			img = transforms.Resize((550, 550))(img)
			img = transforms.CenterCrop(INPUT_SIZE)(img)
			img = transforms.ToTensor()(img)
			img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)

		return img, target

	def __len__(self):
		if self.is_train:
			return len(self.train_label)
		else:
			return len(self.test_label)


class Cars(VisionDataset):
	"""`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset.
	Args:
		root (string): Root directory of the dataset.
		train (bool, optional): If True, creates dataset from training set, otherwise
			creates from test set.
		transform (callable, optional): A function/transform that  takes in an PIL image
			and returns a transformed version. E.g, ``transforms.RandomCrop``
		target_transform (callable, optional): A function/transform that takes in the
			target and transforms it.
		download (bool, optional): If true, downloads the dataset from the internet and
			puts it in root directory. If dataset is already downloaded, it is not
			downloaded again.
	"""
	file_list = {
		'imgs': ('http://imagenet.stanford.edu/internal/car196/car_ims.tgz', 'car_ims'),
		'annos': ('http://imagenet.stanford.edu/internal/car196/cars_annos.mat', 'cars_annos.mat')
	}

	def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
		super(Cars, self).__init__(root, transform=transform, target_transform=target_transform)

		self.loader = default_loader
		self.train = train

		# if self._check_exists():
		# 	print('Files already downloaded and verified.')
		# elif download:
		# 	self._download()
		# else:
		# 	raise RuntimeError(
		# 		'Dataset not found. You can use download=True to download it.')

		loaded_mat = scipy.io.loadmat(os.path.join(self.root, self.file_list['annos'][1]))
		loaded_mat = loaded_mat['annotations'][0]
		self.samples = []
		for item in loaded_mat:
			if self.train != bool(item[-1][0]):
				path = str(item[0][0])
				label = int(item[-2][0]) - 1
				self.samples.append((path, label))

	def __getitem__(self, index):
		path, target = self.samples[index]
		path = os.path.join(self.root, path)

		image = self.loader(path)
		if self.transform is not None:
			image = self.transform(image)
		if self.target_transform is not None:
			target = self.target_transform(target)
		return image, target

	def __len__(self):
		return len(self.samples)

	def _check_exists(self):
		print(os.path.join(self.root, self.file_list['imgs'][1]))
		return os.path.exists(os.path.join(self.root, self.file_list['imgs'][1]))

	def _download(self):
		print('Downloading...')
		for url, filename in self.file_list.values():
			download_url(url, root=self.root, filename=filename)
		print('Extracting...')
		archive = os.path.join(self.root, self.file_list['imgs'][1])
		extract_archive(archive)

class DOG():
	def __init__(self, root, is_train=True, data_len=None):
		self.root = root
		self.is_train = is_train

		if self.is_train:
			data = sio.loadmat(os.path.join(root, 'lists/train_list.mat'))
			self.train_file_list = [x[0][0] for x in data['file_list']]
			self.train_label = [x[0].astype('int64') - 1 for x in data['labels']]
		else:
			data = sio.loadmat(os.path.join(root, 'lists/test_list.mat'))
			self.test_file_list = [x[0][0] for x in data['file_list']]
			self.test_label = [x[0].astype('int64') - 1 for x in data['labels']]

	def __getitem__(self, index):
		if self.is_train:
			img_path = os.path.join(self.root, 'Images', self.train_file_list[index])
			img, target = reader.imread(img_path), self.train_label[index]
			if len(img.shape) == 2:
				img = np.stack([img] * 3, 2)
			img = Image.fromarray(img, mode='RGB')
			# img = transforms.Resize((600, 600), Image.BILINEAR)(img)
			# img = transforms.RandomCrop(INPUT_SIZE)(img)
			# img = transforms.RandomHorizontalFlip()(img)
			img = transforms.Resize((550, 550))(img)
			img = transforms.RandomCrop(INPUT_SIZE, padding=8)(img)
			img = transforms.RandomHorizontalFlip()(img)
			img = transforms.ToTensor()(img)
			# img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
			img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)

		else:
			img_path = os.path.join(self.root, 'Images', self.test_file_list[index])
			img, target = reader.imread(img_path), self.test_label[index]
			if len(img.shape) == 2:
				img = np.stack([img] * 3, 2)
				# img = img[y1:y2, x1:x2, :]
			img = Image.fromarray(img, mode='RGB')
			# img = transforms.Resize((600, 600), Image.BILINEAR)(img)
			# img = transforms.CenterCrop(INPUT_SIZE)(img)
			# img = transforms.ToTensor()(img)
			# img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
			img = transforms.Resize((550, 550))(img)
			img = transforms.CenterCrop(INPUT_SIZE)(img)
			img = transforms.ToTensor()(img)
			img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)

		return img, target

	def __len__(self):
		if self.is_train:
			return len(self.train_label)
		else:
			return len(self.test_label)


class AIR():
	def __init__(self, root, is_train=True, data_len=None):
		self.root = root
		self.is_train = is_train

		variants = []
		with open(os.path.join(root, 'data/variants.txt'), 'r') as f:
			for line in f:
				variants.append(line.strip())

		if self.is_train:
			self.train_file_list = []
			self.train_label = []
			with open(os.path.join(root, 'data/images_variant_trainval.txt'), 'r') as f:
				for line in f:
					line = line.strip()
					seven_digits = line[:7]
					variant = line[8:]
					self.train_file_list.append(os.path.join(root, 'data/images', seven_digits + '.jpg'))
					self.train_label.append(variants.index(variant))
		else:
			self.test_file_list = []
			self.test_label = []
			with open(os.path.join(root, 'data/images_variant_test.txt'), 'r') as f:
				for line in f:
					line = line.strip()
					seven_digits = line[:7]
					variant = line[8:]
					self.test_file_list.append(os.path.join(root, 'data/images', seven_digits + '.jpg'))
					self.test_label.append(variants.index(variant))

	def __getitem__(self, index):
		if self.is_train:
			img_path = self.train_file_list[index]
			img, target = reader.imread(img_path)[:-20, ...], self.train_label[index]
			if len(img.shape) == 2:
				img = np.stack([img] * 3, 2)
			img = Image.fromarray(img, mode='RGB')
			# img = transforms.Resize((600, 600), Image.BILINEAR)(img)
			# img = transforms.RandomCrop(INPUT_SIZE)(img)
			# img = transforms.RandomHorizontalFlip()(img)
			img = transforms.Resize((550, 550))(img)
			img = transforms.RandomCrop(INPUT_SIZE, padding=8)(img)
			img = transforms.RandomHorizontalFlip()(img)
			img = transforms.ToTensor()(img)
			# img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
			img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)

		else:
			img_path = self.test_file_list[index]
			img, target = reader.imread(img_path)[:-20, ...], self.test_label[index]
			if len(img.shape) == 2:
				img = np.stack([img] * 3, 2)
				# img = img[y1:y2, x1:x2, :]
			img = Image.fromarray(img, mode='RGB')
			# img = transforms.Resize((600, 600), Image.BILINEAR)(img)
			# img = transforms.CenterCrop(INPUT_SIZE)(img)
			# img = transforms.ToTensor()(img)
			# img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
			img = transforms.Resize((550, 550))(img)
			img = transforms.CenterCrop(INPUT_SIZE)(img)
			img = transforms.ToTensor()(img)
			img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)

		return img, target

	def __len__(self):
		if self.is_train:
			return len(self.train_label)
		else:
			return len(self.test_label)


if __name__ == '__main__':
	dataset = AIR(root='E:/Datasets/fgvc-aircraft-2013b/data')
	print(len(dataset.train_file_list))
	print(len(dataset.train_label))
	for data in dataset:
		print(data[0].size(), data[1])
