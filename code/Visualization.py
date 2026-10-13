### Visualization ###

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

repo_path = Path(__file__).resolve().parent.parent

metric_path = repo_path / "results_full" / "metrics"             # Change to correct path #
output_path = repo_path / "results_full" / "Visualizations"      # Change to correct path #

output_path.mkdir(parents=True, exist_ok=True)

model_csv_list = [
    #metric_path / "n28000_bs8_ep50_random_training_history.csv",
    #metric_path / "n28000_bs8_ep50_34_33_33_training_history.csv",
    metric_path / "n28000_bs8_ep50_10_80_10_training_history.csv",
    #metric_path / "dpt_n28000_bs8_ep50_random_DPT_training_history.csv",
    #metric_path / "dpt_n28000_bs8_ep50_34_33_33_DPT_training_history.csv",
    #metric_path / "dpt_n28000_bs8_ep50_10_80_10_DPT_training_history.csv",
    #metric_path / "dino_resnet50_n28000_bs8_ep50_random_DINO_training_history.csv",
    #metric_path / "dino_resnet50_n28000_bs8_ep50_34_33_33_DINO_training_history.csv",
    #metric_path / "dino_resnet50_n28000_bs8_ep50_10_80_10_DINO_training_history.csv",
    #metric_path / "n28000_bs8_ep50_lr0.001_10_80_10_training_history.csv",
    #metric_path / "n28000_bs8_ep50_lr1e-05_10_80_10_training_history.csv",
    metric_path / "n28000_bs8_ep50_lr0.0001_wd0.001_10_80_10_training_history.csv",
    metric_path / "n28000_bs8_ep50_lr0.0001_wd1e-05_10_80_10_training_history.csv",
]

model_metric_csv_list = [
    #metric_path / "n28000_bs8_ep50_random_training_summary.csv",
    #metric_path / "n28000_bs8_ep50_34_33_33_training_summary.csv",
    metric_path / "n28000_bs8_ep50_10_80_10_training_summary.csv",
    #metric_path / "dpt_n28000_bs8_ep50_random_DPT_training_summary.csv",
    #metric_path / "dpt_n28000_bs8_ep50_34_33_33_DPT_training_summary.csv",
    #metric_path / "dpt_n28000_bs8_ep50_10_80_10_DPT_training_summary.csv",
    #metric_path / "dino_resnet50_n28000_bs8_ep50_random_DINO_training_summary.csv",
    #metric_path / "dino_resnet50_n28000_bs8_ep50_34_33_33_DINO_training_summary.csv",
    #metric_path / "dino_resnet50_n28000_bs8_ep50_10_80_10_DINO_training_summary.csv",
    #metric_path / "n28000_bs8_ep50_lr0.001_10_80_10_training_summary.csv",
    #metric_path / "n28000_bs8_ep50_lr1e-05_10_80_10_training_summary.csv",
    metric_path / "n28000_bs8_ep50_lr0.0001_wd0.001_10_80_10_training_summary.csv",
    metric_path / "n28000_bs8_ep50_lr0.0001_wd1e-05_10_80_10_training_summary.csv",
]

model_label_list = [
    #"Model_1_random",
    #"Model_1_equal",
    "Model_1_edge_LR=1e-4",
    #"Model_2_random",
    #"Model_2_equal",
    #"Model_2_edge",
    #"Model_3_random",
    #"Model_3_equal",
    #"Model_3_edge",
    #"Model_1_edge_LR=1e-3",
    #"Model_1_edge_LR=1e-5",
    "Model_1_edge_WD=1e-3",
    "Model_1_edge_WD=1e-5",
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

    # Give each model its own color
    colors = plt.cm.tab10.colors

    for i, (csv_path, model_label) in enumerate(
        zip(model_csv_list, model_label_list)
    ):
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

        # Dice
        dice_ax.plot(
            data["epoch"],
            data["train_dice"],
            label=f"{model_label} - Train",
            color=color,
            linestyle="-",
        )
        dice_ax.plot(
            data["epoch"],
            data["val_dice"],
            label=f"{model_label} - Validation",
            color=color,
            linestyle="--",
        )

        # Loss
        loss_ax.plot(
            data["epoch"],
            data["train_loss"],
            label=f"{model_label} - Train",
            color=color,
            linestyle="-",
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

def barplot_model_metrics(
    model_metric_csv_list,
    model_label_list,
    metrics=None,
):
    """
    Creates a grouped bar plot comparing model metrics.

    X-axis:
        Background, Edge, Foreground

    Bars:
        One bar per model for each metric.

    Colors:
        Represent models.
    """

    if len(model_metric_csv_list) != len(model_label_list):
        raise ValueError(
            "model_metric_csv_list and model_label_list must have the same length."
        )

    if not model_metric_csv_list:
        raise ValueError("No model summary CSV files were provided.")

    if metrics is None:
        metrics = [
            "best_val_dice_background",
            "best_val_dice_edge",
            "best_val_dice_foreground",
        ]

    metric_labels = {
        "best_val_dice_background": "Background",
        "best_val_dice_edge": "Edge",
        "best_val_dice_foreground": "Foreground",
    }

    all_data = []

    for csv_path, model_label in zip(
        model_metric_csv_list,
        model_label_list,
    ):
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {csv_path}"
            )

        data = pd.read_csv(csv_path)

        if data.empty:
            raise ValueError(
                f"CSV file is empty: {csv_path}"
            )

        missing_columns = [
            metric
            for metric in metrics
            if metric not in data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns in {csv_path}: {missing_columns}"
            )

        row = data.iloc[0]

        model_data = {
            "model": model_label,
        }

        for metric in metrics:
            model_data[metric] = row[metric]

        all_data.append(model_data)

    data = pd.DataFrame(all_data)

    # ---------------------------------------------------------
    # Create grouped bar plot
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 7))

    x = range(len(metrics))

    n_models = len(data)

    total_width = 0.8
    bar_width = total_width / n_models

    # One color per model
    colors = plt.cm.tab10.colors

    for model_index, (_, row) in enumerate(data.iterrows()):

        offset = (
            model_index * bar_width
            - total_width / 2
            + bar_width / 2
        )

        positions = [
            position + offset
            for position in x
        ]

        values = [
            row[metric]
            for metric in metrics
        ]

        bars = ax.bar(
            positions,
            values,
            width=bar_width,
            label=row["model"],
            color=colors[model_index % len(colors)],
        )

        # Add horizontal/readable values
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=0,
            )

    # ---------------------------------------------------------
    # Axis labels
    # ---------------------------------------------------------

    ax.set_xticks(list(x))

    ax.set_xticklabels(
        [
            metric_labels.get(metric, metric)
            for metric in metrics
        ],
        rotation=0,
    )

    ax.set_ylabel("Dice Score")
    ax.set_xlabel("Region")

    ax.set_title("Model Validation Dice by Region")

    ax.set_ylim(0, 1.0)

    ax.grid(
        axis="y",
        alpha=0.3,
        linestyle="--",
    )

    ax.legend(
        title="Model",
        loc="best",
    )

    fig.tight_layout()

    model_name = "_vs_".join(model_label_list)

    output_file = (
        output_path
        / f"{model_name}_validation_metrics.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved visualization to: {output_file}"
    )

    return output_file


if __name__ == "__main__":
    combine_graph(model_csv_list, model_label_list)
    barplot_model_metrics(
        model_metric_csv_list,
        model_label_list,
        metrics=[
            "best_val_dice_background",
            "best_val_dice_edge",
            "best_val_dice_foreground",
        ],
    )