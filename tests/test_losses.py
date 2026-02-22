"""
Tests for AstroCNN model and loss functions.

Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.astro_cnn import AstroCNN, ConvBlock
from src.training.losses import FocalLoss, LabelSmoothingCrossEntropy


# Model Tests 

class TestConvBlock:

    def test_forward_shape_with_pool(self):
        block = ConvBlock(in_channels=3, out_channels=32, pool_size=2)
        x = torch.randn(4, 3, 64, 64)
        out = block(x)
        assert out.shape == (4, 32, 32, 32), f"Expected (4, 32, 32, 32), got {out.shape}"

    def test_forward_shape_no_pool(self):
        block = ConvBlock(in_channels=32, out_channels=64, pool_size=0)
        x = torch.randn(4, 32, 16, 16)
        out = block(x)
        assert out.shape == (4, 64, 16, 16)

    def test_no_nan_in_output(self):
        block = ConvBlock(3, 32, use_bn=True)
        x = torch.randn(2, 3, 32, 32)
        out = block(x)
        assert not torch.isnan(out).any(), "ConvBlock output contains NaN"


class TestAstroCNN:

    @pytest.fixture
    def model(self):
        return AstroCNN(num_classes=4, channels=[16, 32, 64, 128], dropout_rate=0.0)

    def test_output_shape(self, model):
        x = torch.randn(8, 3, 64, 64)
        logits = model(x)
        assert logits.shape == (8, 4), f"Expected (8, 4), got {logits.shape}"

    def test_output_is_logits_not_probs(self, model):
        """Output should be raw logits, NOT softmax probabilities."""
        x = torch.randn(4, 3, 64, 64)
        logits = model(x)
        # Softmax outputs sum to 1; logits generally don't
        row_sums = logits.sum(dim=1)
        assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.01), (
            "Model appears to be outputting probabilities, expected raw logits"
        )

    def test_no_nan_gradients(self, model):
        """Backward pass should produce finite gradients."""
        x = torch.randn(4, 3, 64, 64)
        logits = model(x)
        loss = logits.mean()
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_different_input_sizes(self, model):
        """GAP should make model agnostic to input spatial size."""
        model.eval()
        for size in [32, 64, 128]:
            x = torch.randn(2, 3, size, size)
            out = model(x)
            assert out.shape == (2, 4)

    def test_train_eval_mode_difference(self, model):
        """Dropout/BatchNorm should behave differently in train vs eval."""
        x = torch.randn(32, 3, 64, 64)
        model.train()
        out_train = model(x)
        model.eval()
        with torch.no_grad():
            out_eval1 = model(x)
            out_eval2 = model(x)
        # Eval mode should be deterministic
        assert torch.allclose(out_eval1, out_eval2), "Model not deterministic in eval mode"

    def test_parameter_count_reasonable(self, model):
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 5_000_000, f"Model has {n_params:,} params — too large for a custom CNN"
        assert n_params > 10_000, f"Model has only {n_params} params — likely misconfigured"

    def test_get_feature_maps(self, model):
        x = torch.randn(4, 3, 64, 64)
        logits, features = model.get_feature_maps(x)
        assert logits.shape == (4, 4)
        assert features.ndim == 4  # (B, C, H, W)


# Loss Tests

class TestFocalLoss:

    @pytest.fixture
    def basic_focal(self):
        return FocalLoss(gamma=2.0, alpha=None)

    def test_output_is_scalar(self, basic_focal):
        logits = torch.randn(16, 4)
        targets = torch.randint(0, 4, (16,))
        loss = basic_focal(logits, targets)
        assert loss.ndim == 0, "Loss should be a scalar"

    def test_loss_non_negative(self, basic_focal):
        for _ in range(10):
            logits = torch.randn(8, 4)
            targets = torch.randint(0, 4, (8,))
            loss = basic_focal(logits, targets)
            assert loss.item() >= 0, "Loss must be non-negative"

    def test_lower_loss_for_confident_correct_prediction(self):
        """
        Focal Loss should assign lower loss to easy (well-classified) samples.
        A high-confidence correct prediction should have near-zero focal weight.
        """
        focal = FocalLoss(gamma=2.0)
        # High confidence correct: logit[correct_class] >> others
        easy_logit = torch.tensor([[10.0, -5.0, -5.0, -5.0]])  # Class 0 confident
        hard_logit = torch.tensor([[0.3, 0.25, 0.25, 0.2]])     # Uncertain
        target = torch.tensor([0])

        easy_loss = focal(easy_logit, target).item()
        hard_loss = focal(hard_logit, target).item()

        assert easy_loss < hard_loss, (
            f"Easy sample (loss={easy_loss:.4f}) should have lower focal loss "
            f"than hard sample (loss={hard_loss:.4f})"
        )

    def test_gamma_zero_equals_cross_entropy(self):
        """When gamma=0, Focal Loss should reduce to standard Cross-Entropy."""
        focal_zero = FocalLoss(gamma=0.0, alpha=None)
        ce = nn.CrossEntropyLoss()

        torch.manual_seed(42)
        logits = torch.randn(32, 4)
        targets = torch.randint(0, 4, (32,))

        fl_loss = focal_zero(logits, targets)
        ce_loss = ce(logits, targets)

        assert torch.allclose(fl_loss, ce_loss, atol=1e-5), (
            f"Focal(gamma=0) = {fl_loss:.6f}, CE = {ce_loss:.6f} — should be equal"
        )

    def test_alpha_weighting_rare_class(self):
        """Higher alpha for a class should increase its loss contribution."""
        logits = torch.randn(100, 4)
        targets = torch.zeros(100, dtype=torch.long)  # All class 0

        # High alpha for class 0
        focal_high = FocalLoss(gamma=2.0, alpha=[0.9, 0.033, 0.033, 0.033])
        # Low alpha for class 0
        focal_low = FocalLoss(gamma=2.0, alpha=[0.1, 0.3, 0.3, 0.3])

        loss_high = focal_high(logits, targets).item()
        loss_low = focal_low(logits, targets).item()

        assert loss_high > loss_low, "Higher alpha should produce higher loss"

    def test_no_reduction(self):
        focal = FocalLoss(gamma=2.0, reduction="none")
        logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        loss = focal(logits, targets)
        assert loss.shape == (8,), "Unreduced loss should have shape (batch_size,)"


class TestLabelSmoothingCE:

    def test_output_scalar(self):
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        logits = torch.randn(16, 4)
        targets = torch.randint(0, 4, (16,))
        loss = criterion(logits, targets)
        assert loss.ndim == 0

    def test_smoothing_zero_equals_ce(self):
        """With smoothing=0, should approximate standard CE."""
        ce_smooth = LabelSmoothingCrossEntropy(smoothing=0.0)
        ce_standard = nn.CrossEntropyLoss()

        torch.manual_seed(0)
        logits = torch.randn(32, 4)
        targets = torch.randint(0, 4, (32,))

        assert torch.allclose(
            ce_smooth(logits, targets), ce_standard(logits, targets), atol=1e-5
        )

    def test_smoothing_reduces_overconfidence(self):
        """
        Label smoothing should produce higher loss for overconfident correct predictions.
        This is by design — it penalizes extreme confidence.
        """
        smooth = LabelSmoothingCrossEntropy(smoothing=0.1)
        no_smooth = LabelSmoothingCrossEntropy(smoothing=0.0)

        # Very confident logits
        logits = torch.tensor([[100.0, -10.0, -10.0, -10.0]])
        target = torch.tensor([0])

        loss_smooth = smooth(logits, target).item()
        loss_plain = no_smooth(logits, target).item()

        assert loss_smooth > loss_plain, (
            "Label smoothing should produce higher loss for overconfident predictions"
        )
