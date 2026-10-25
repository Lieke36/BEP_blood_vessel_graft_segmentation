### Losses and metrics ###

# What does Losses_metrics.py do?
# This script contain the functions to calculate the metrics of the models and the loss functions. 
# The loss function has a Dice weight of 0.5 and a BCE weight of 0.5.

# How to use Losses_metrics.py?
# No actions are needed before using this code. The script is called in the model and evaluation scripts.

## loss functions ##
# Dice loss
# BCE loss


## metrics ##
# Dice
# sensitivity
# recall


## libraries ##
import torch
import torch.nn as nn

## Dice loss ##

class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.

    The model output should contain raw logits.
    Sigmoid is applied inside this loss.
    """

    def __init__(self, smooth=1.0):
        super().__init__()

        self.smooth = smooth

    def forward(self, logits, targets):

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.view(
            probabilities.size(0),
            -1
        )

        targets = targets.view(
            targets.size(0),
            -1
        )

        intersection = (
            probabilities * targets
        ).sum(dim=1)

        denominator = (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
        )

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            denominator
            + self.smooth
        )

        dice = dice.mean()

        return 1.0 - dice


## Combined BCE + Dice loss ##

class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy + Dice loss.

        loss = 0.5 * BCE + 0.5 * Dice

    BCEWithLogitsLoss is used because the model outputs logits.
    """

    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5,
        smooth=1.0,
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss(
            smooth=smooth
        )

    def forward(self, logits, targets):

        bce_loss = self.bce(
            logits,
            targets
        )

        dice_loss = self.dice(
            logits,
            targets
        )

        total_loss = (
            self.bce_weight * bce_loss
            +
            self.dice_weight * dice_loss
        )

        return total_loss


## Dice metric ##

def dice_score(
    logits,
    targets,
    threshold=0.5,
    smooth=1.0,
):
    """
    Calculates binary Dice score.
    """

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= threshold
    ).float()

    predictions = predictions.view(
        predictions.size(0),
        -1
    )

    targets = targets.view(
        targets.size(0),
        -1
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    denominator = (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
    )

    dice = (
        2.0 * intersection
        + smooth
    ) / (
        denominator
        + smooth
    )

    return dice.mean().item()


## Sensitivity / Recall ##

def sensitivity_score(
    logits,
    targets,
    threshold=0.5,
    smooth=1e-7,
):
    """
    Calculates foreground sensitivity / recall.

        sensitivity = TP / (TP + FN)
    """

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= threshold
    ).float()

    predictions = predictions.view(
        predictions.size(0),
        -1
    )

    targets = targets.view(
        targets.size(0),
        -1
    )

    true_positive = (
        predictions * targets
    ).sum(dim=1)

    false_negative = (
        (1.0 - predictions) * targets
    ).sum(dim=1)

    sensitivity = (
        true_positive
        /
        (
            true_positive
            + false_negative
            + smooth
        )
    )

    return sensitivity.mean().item()


## Precision ##

def precision_score(
    logits,
    targets,
    threshold=0.5,
    smooth=1e-7,
):
    """
    Calculates foreground precision.

        precision = TP / (TP + FP)
    """

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= threshold
    ).float()

    predictions = predictions.view(
        predictions.size(0),
        -1
    )

    targets = targets.view(
        targets.size(0),
        -1
    )

    true_positive = (
        predictions * targets
    ).sum(dim=1)

    false_positive = (
        predictions
        * (1.0 - targets)
    ).sum(dim=1)

    precision = (
        true_positive
        /
        (
            true_positive
            + false_positive
            + smooth
        )
    )

    return precision.mean().item()


## Calculate all metrics ##

def calculate_metrics(
    logits,
    targets,
    threshold=0.5,
):
    """
    Calculates the segmentation metrics currently used.

    Returns:
        dice
        sensitivity
        precision
    """

    return {
        "dice": dice_score(
            logits,
            targets,
            threshold=threshold,
        ),

        "sensitivity": sensitivity_score(
            logits,
            targets,
            threshold=threshold,
        ),

        "precision": precision_score(
            logits,
            targets,
            threshold=threshold,
        ),
    }
