"""
Tests for AstroDataset and DataLoader pipeline.

Run with: pytest tests/test_dataset.py -v
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import AstroDataset, build_train_transforms, build_val_transforms


def create_dummy_dataset(root: Path, classes: list, images_per_class: dict) -> None:
    """Create a minimal on-disk dataset structure for testing."""
    for cls in classes:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        n = images_per_class.get(cls, 10)
        for i in range(n):
            # Create a random 64x64 RGB image
            arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(arr, mode="RGB")
            img.save(cls_dir / f"img_{i:04d}.jpg")


CLASSES = ["star", "galaxy", "quasar", "supernova"]
IMAGES_PER_CLASS = {"star": 100, "galaxy": 20, "quasar": 5, "supernova": 2}


@pytest.fixture(scope="module")
def dummy_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("astro_data")
    create_dummy_dataset(root, CLASSES, IMAGES_PER_CLASS)
    return root


class TestAstroDataset:

    def test_length(self, dummy_root):
        dataset = AstroDataset(root_dir=dummy_root, classes=CLASSES)
        expected = sum(IMAGES_PER_CLASS.values())
        assert len(dataset) == expected, f"Expected {expected} samples, got {len(dataset)}"

    def test_class_counts(self, dummy_root):
        dataset = AstroDataset(root_dir=dummy_root, classes=CLASSES)
        for cls, expected_count in IMAGES_PER_CLASS.items():
            assert dataset.class_counts[cls] == expected_count

    def test_getitem_returns_tensor_and_label(self, dummy_root):
        import torchvision.transforms as T
        transform = T.Compose([T.Resize((32, 32)), T.ToTensor()])
        dataset = AstroDataset(root_dir=dummy_root, classes=CLASSES, transform=transform)
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 32, 32)
        assert isinstance(label, int)
        assert 0 <= label < len(CLASSES)

    def test_label_encoding(self, dummy_root):
        dataset = AstroDataset(root_dir=dummy_root, classes=CLASSES)
        assert dataset.class_to_idx["star"] == 0
        assert dataset.class_to_idx["supernova"] == 3
        assert dataset.idx_to_class[0] == "star"

    def test_sample_weights_shape(self, dummy_root):
        dataset = AstroDataset(root_dir=dummy_root, classes=CLASSES)
        weights = dataset.get_sample_weights()
        assert weights.shape == (len(dataset),)
        assert (weights > 0).all()

    def test_sample_weights_inverse_frequency(self, dummy_root):
        """Rare classes should have higher sample weights."""
        dataset = AstroDataset(root_dir=dummy_root, classes=CLASSES)
        weights = dataset.get_sample_weights()

        # Build mapping: sample index → class label
        star_weights = [w for (_, label), w in zip(dataset.samples, weights) if label == 0]
        supernova_weights = [w for (_, label), w in zip(dataset.samples, weights) if label == 3]

        assert supernova_weights[0] > star_weights[0], (
            "Supernova (rare) should have higher sample weight than star (common)"
        )

    def test_return_path(self, dummy_root):
        import torchvision.transforms as T
        transform = T.Compose([T.Resize((32, 32)), T.ToTensor()])
        dataset = AstroDataset(
            root_dir=dummy_root, classes=CLASSES, transform=transform, return_path=True
        )
        image, label, path = dataset[0]
        assert isinstance(path, str)
        assert Path(path).exists()

    def test_missing_class_dir(self, tmp_path):
        """Dataset should warn and skip missing class directories, not crash."""
        (tmp_path / "star").mkdir()
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(arr).save(tmp_path / "star" / "img.jpg")
        # galaxy directory missing — should log warning but continue
        dataset = AstroDataset(root_dir=tmp_path, classes=["star", "galaxy"])
        assert len(dataset) > 0  # star images still loaded

    def test_nonexistent_root_raises(self):
        with pytest.raises(FileNotFoundError):
            AstroDataset(root_dir="/nonexistent/path", classes=CLASSES)


class TestTransforms:

    def test_val_transform_deterministic(self, dummy_root):
        """Same image should produce identical tensors under val transforms (no randomness)."""
        import torchvision.transforms as T

        class _MockConfig:
            class data:
                image_size = 64
            class augmentation:
                class val:
                    class normalize:
                        mean = [0.5, 0.5, 0.5]
                        std = [0.5, 0.5, 0.5]

        transform = build_val_transforms(_MockConfig())
        dataset = AstroDataset(root_dir=dummy_root, classes=CLASSES, transform=transform)
        img1, _ = dataset[0]
        img2, _ = dataset[0]
        assert torch.allclose(img1, img2), "Val transforms must be deterministic"
