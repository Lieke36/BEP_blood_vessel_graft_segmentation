### Visualization ###

# 1. Gets metric folder path
# 2. Gets list of .csv files with names
# 3. Makes 2 graphs of epochs vs val/train dice and epochs vs val/train loss
# 4. Saves graphs to png in folder Visualizations with name of models

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

repo_path = Path(__file__).resolve().parent.parent

metric_path = repo_path / "results_full" / "metrics"             # Change to correct path #
output_path = repo_path / "results_full" / "visualizations"      # Change to correct path #

output_path.mkdir(parents=True, exist_ok=True)

model_csv_list = [
    #metric_path / "n10000_bs8_ep50_34_33_33_training_history.csv",
    #metric_path / "n10000_bs8_ep50_training_history.csv",
    #metric_path / "dpt_n10000_bs8_ep50_34_33_33_DPT_training_history.csv",
    #metric_path / "dpt_n10000_bs8_ep50_DPT_training_history.csv",
    metric_path / "dino_resnet50_n10000_bs8_ep50_34_33_33_DINO_training_history.csv",
    metric_path / "dino_resnet50_n10000_bs8_ep50_DINO_training_history.csv",
]

model_label_list = [
    #"Model_0_equal",
    #"Model_0_random",
    #"Model_1_equal",
    #"Model_1_random",
    "Model_2_equal",
    "Model_2_random",
]

def combine_graph(model_csv_list, model_label_list):
    """Combines training histories from multiple models into one PNG."""
    if len(model_csv_list) != len(model_label_list):
        raise ValueError("model_csv_list and model_label_list must have the same length.")
    if not model_csv_list:
        raise ValueError("No model CSV files were provided.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    dice_ax, loss_ax = axes
    model_names = []

    required_columns = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_dice",
        "val_dice",
        "val_sensitivity",
        "val_precision",
    ]

    colors = plt.cm.tab10.colors

    for i, (csv_path, model_label) in enumerate(zip(model_csv_list, model_label_list)):
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        data = pd.read_csv(csv_path)

        missing_columns = [
            column for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns in {csv_path}: {missing_columns}"
            )

        color = colors[i % len(colors)]

        dice_ax.plot(
            data["epoch"],
            data["train_dice"],
            label=f"{model_label} - Train",
            color=color,
        )
        dice_ax.plot(
            data["epoch"],
            data["val_dice"],
            label=f"{model_label} - Validation",
            color=color,
            linestyle="--",
        )

        loss_ax.plot(
            data["epoch"],
            data["train_loss"],
            label=f"{model_label} - Train",
            color=color,
        )
        loss_ax.plot(
            data["epoch"],
            data["val_loss"],
            label=f"{model_label} - Validation",
            color=color,
            linestyle="--",
        )

        model_names.append(model_label)

    dice_ax.set_title("Training and Validation Dice")
    dice_ax.set_xlabel("Epoch")
    dice_ax.set_ylabel("Dice")
    dice_ax.grid(True, alpha=0.3)
    dice_ax.legend()

    loss_ax.set_title("Training and Validation Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.grid(True, alpha=0.3)
    loss_ax.legend()

    fig.suptitle("Model Training History", fontsize=16)
    fig.tight_layout()

    model_name = "_vs_".join(model_names)
    output_file = output_path / f"{model_name}_training_history.png"

    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved visualization to: {output_file}")
    return output_file

if __name__ == "__main__":
    combine_graph(model_csv_list, model_label_list)