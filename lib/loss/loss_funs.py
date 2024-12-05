import numpy as np
import torch
from monai.losses import DiceLoss, GeneralizedDiceLoss, DiceCELoss, FocalLoss
from monai.networks import one_hot
import torch.nn as nn
import torch.nn.functional as F

class diceCELoss(nn.Module):
    def __init__(
        self, 
        include_background=True, 
        to_onehot_y=False,
        softmax=False, 
        sigmoid=True, 
        ce_weight=None, 
        reduction='mean', 
        smooth_nr=1e-6, 
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0):
        super(diceCELoss, self).__init__()

        self.loss_fn = DiceCELoss(include_background=include_background, 
                                  to_onehot_y=to_onehot_y,
                                  softmax=softmax,
                                  sigmoid=sigmoid, 
                                  smooth_nr=smooth_nr, 
                                  reduction=reduction,
                                  ce_weight=ce_weight,
                                  lambda_dice=lambda_dice,
                                  lambda_ce=lambda_ce)
                                  
    def forward(self, output, target):
        return self.loss_fn(output, target)

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        
        self.BCELoss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, output, target):
        bce_loss = self.BCELoss_fn(output, target)
        return bce_loss


class diceLoss(nn.Module):
    def __init__(self, 
        include_background=True, 
        to_onehot_y=True, 
        softmax=False, 
        sigmoid=True, 
        ce_weight=None, 
        reduction='mean', 
        smooth_nr=1e-6):

        super(diceLoss, self).__init__()
        self.DiceLoss_fn = DiceLoss(include_background=include_background,
                                    to_onehot_y=to_onehot_y,
                                    softmax=softmax,
                                    sigmoid=sigmoid,
                                    reduction=reduction, 
                                    smooth_dr=smooth_nr)

    def forward(self, output, target):
        dice_loss = self.DiceLoss_fn(output, target)
        return dice_loss
    

class focalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self, gamma=1.5, alpha=0.25):
        """Initializer for FocalLoss class with no parameters."""
        super(focalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, output, target):
        loss = F.binary_cross_entropy_with_logits(output, target, reduction='none')
        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        output_prob = output.sigmoid()  # prob from logits
        p_t = target * output_prob + (1 - target) * (1 - output_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if self.alpha > 0:
            alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)
            loss *= alpha_factor
        reduce_axis: list[int] = torch.arange(2, len(output.shape)).tolist()
        return torch.mean(loss, reduce_axis).mean()


class generalizedDiceLoss(nn.Module):
    def __init__(
        self, 
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        smooth: float = 1e-5,
        ):

        super(generalizedDiceLoss, self).__init__()
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.smooth = float(smooth)
    
    def forward(self, output, target):
        
        n_pred_ch = output.shape[1]

        if self.sigmoid:
            output = torch.sigmoid(output)
        
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                output = torch.softmax(output, 1)
        
        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)
        
        if not self.include_background:
            if n_output_ch == 1:
                warnings.warn("single channel outputiction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                output = output[:, 1:]

        reduce_axis: list[int] = torch.arange(2, len(output.shape)).tolist()

        wei = torch.sum(target, axis=reduce_axis) # (n_class,)
        wei = torch.reciprocal(wei**2+self.smooth)

        ground_o = torch.sum(target, reduce_axis)
        output_o = torch.sum(output, reduce_axis)

        denominator = ground_o + output_o

        intersection = wei*torch.sum(output * target, axis=reduce_axis)
        union = wei*denominator
        gldice_loss = 1 - (2. * intersection) / (union + self.smooth)
        return torch.mean(gldice_loss)
    
class generalizedDiceFocalLoss(nn.Module):
    def __init__(self, include_background=True, sigmoid=True, reduction='mean', gamma=1.5,alpha=0.25, smooth=1e-5, lambda_gd=0.5, lambda_fd=2):
        super(generalizedDiceFocalLoss, self).__init__()
        self.focalLoss = focalLoss(gamma=gamma, alpha=alpha)
        self.gldiceLoss = generalizedDiceLoss(include_background=include_background, sigmoid=sigmoid)
        self.lambda_fd = lambda_fd
        self.lambda_gd = lambda_gd

    def forward(self, output, target):
        return self.lambda_fd * self.focalLoss(output, target) + self.lambda_gd * self.gldiceLoss(output, target)

class generalizedDiceCELoss(nn.Module):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = True,
        sigmoid: bool = False,
        softmax: bool = True,
        reduction: str = "mean",
        smooth: float = 1e-5,
        batch: bool = False,
        ce_weight = None,
        lambda_gdice: float = 0.5,
        lambda_ce: float = 0.5,
    ):
        super(generalizedDiceCELoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction=reduction)
        self.gldiceLoss = generalizedDiceLoss(
                                        include_background = include_background,
                                        to_onehot_y = to_onehot_y,
                                        sigmoid = sigmoid,
                                        softmax = softmax,
                                        smooth = smooth)
        self.lambda_ce = lambda_ce
        self.lambda_gd = lambda_gdice
    
    def ce(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute CrossEntropy loss for the input and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
        elif self.old_pt_ver:
            warnings.warn(
                f"Multichannel targets are not supported in this older Pytorch version {torch.__version__}. "
                "Using argmax (as a workaround) to convert target to a single channel."
            )
            target = torch.argmax(target, dim=1)
        elif not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        return self.cross_entropy(input, target)  # type: ignore[no-any-return]

    def forward(self, output, target):
        reduce_axis: list[int] = torch.arange(2, len(output.shape)).tolist()
        ce_loss = self.ce(output, target)
        gdice_loss = self.gldiceLoss(output, target)
        loss = self.lambda_ce * ce_loss + self.lambda_gd * gdice_loss

        return loss


class WeightDiceCELoss(nn.Module):
    """
    Compute both Dice loss and Cross Entropy Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Cross Entropy Loss is shown in ``torch.nn.CrossEntropyLoss``. In this implementation,
    two deprecated parameters ``size_average`` and ``reduce``, and the parameter ``ignore_index`` are
    not supported.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = True,
        sigmoid: bool = True,
        softmax: bool = False,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ) -> None:
        """
        Args:
            ``ce_weight`` and ``lambda_ce`` are only used for cross entropy loss.
            ``reduction`` is used for both losses and other parameters are only used for dice loss.

            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss`.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``. only used by the `DiceLoss`, not for the `CrossEntropyLoss`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``. The dice loss should
                as least reduce the spatial dimensions, which is different from cross entropy loss, thus here
                the ``none`` option cannot be used.

                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            ce_weight: a rescaling weight given to each class for cross entropy loss.
                See ``torch.nn.CrossEntropyLoss()`` for more information.
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_ce: the trade-off weight value for cross entropy loss. The value should be no less than 0.0.
                Defaults to 1.0.

        """
        super().__init__()
        reduction = look_up_option(reduction, DiceCEReduction).value
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction=reduction)
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.old_pt_ver = not pytorch_after(1, 10)

    def ce(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute CrossEntropy loss for the input and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
        elif self.old_pt_ver:
            warnings.warn(
                f"Multichannel targets are not supported in this older Pytorch version {torch.__version__}. "
                "Using argmax (as a workaround) to convert target to a single channel."
            )
            target = torch.argmax(target, dim=1)
        elif not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        return self.cross_entropy(input, target)  # type: ignore[no-any-return]

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )

        dice_loss = self.dice(input, target)
        ce_loss = self.ce(input, target)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss

        return total_loss