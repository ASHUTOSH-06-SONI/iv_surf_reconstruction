"""Command-line interface for PINN training and evaluation."""

import logging
import click
from pathlib import Path

from src.config import load_config
from src.data.preprocess import prepare_processed_data
from src.train.train_and_save import train_model


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """IV Surface Reconstruction - PINN Model Tools."""
    pass


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    default="config/default.yaml",
    help="Path to configuration YAML file"
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"]),
    default=None,
    help="Compute device (cpu or cuda)"
)
def train(config: str, device: str):
    """Train PINN model with physics constraints.
    
    Example:
        pinn-train --config config/default.yaml --device cuda
    """
    logger.info(f"Loading configuration from {config}")
    cfg = load_config(config)
    
    # Override device if specified
    if device:
        cfg.training.device = device
    
    logger.info(f"Training with config: {config}")
    logger.info(f"Device: {cfg.training.device}")
    logger.info(f"Model: {cfg.model.name}")
    logger.info(f"Hidden dims: {cfg.model.hidden_dims}")
    logger.info(f"Epochs: {cfg.training.epochs}")
    
    try:
        history = train_model(cfg, save_best_model=True)
        logger.info("Training completed successfully")
        logger.info(f"Best model saved to: {cfg.training.best_model_path}")
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    default="config/default.yaml",
    help="Path to configuration YAML file"
)
def prepare_data(config: str):
    """Prepare processed training CSV from raw parquet files."""
    logger.info(f"Loading configuration from {config}")
    cfg = load_config(config)
    raw_path = getattr(cfg.data, "raw_train_file", None)
    output_file = cfg.data.train_file
    if not raw_path:
        logger.error("No raw_train_file configured. Please update config/default.yaml")
        return 1
    if not Path(raw_path).exists():
        logger.error(f"Raw training file not found: {raw_path}")
        return 1
    try:
        prepare_processed_data(raw_train_file=raw_path, output_train_file=output_file)
        logger.info(f"Processed training data saved to {output_file}")
        return 0
    except Exception as e:
        logger.error(f"Failed to prepare data: {e}", exc_info=True)
        return 1


@cli.command()
@click.option(
    "--model",
    type=click.Path(exists=True),
    required=True,
    help="Path to trained model checkpoint"
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    default="config/default.yaml",
    help="Path to configuration YAML file"
)
@click.option(
    "--output",
    type=click.Path(),
    default="results/evaluation.txt",
    help="Path to save evaluation results"
)
def evaluate(model: str, config: str, output: str):
    """Evaluate trained PINN model on validation/test set.
    
    Example:
        pinn-evaluate --model models/best_model.pt --config config/default.yaml
    """
    logger.info(f"Loading model from {model}")
    logger.info(f"Configuration: {config}")
    
    # TODO: Implement evaluation logic
    logger.info("Evaluation functionality coming soon")
    
    return 0


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    default="config/default.yaml",
    help="Path to configuration YAML file"
)
@click.option(
    "--model",
    type=click.Path(exists=True),
    default="models/best_model.pt",
    help="Path to trained PINN model"
)
@click.option(
    "--output",
    type=click.Path(),
    default="results/comparison.csv",
    help="Path to save comparison results"
)
def compare(config: str, model: str, output: str):
    """Compare PINN with baseline methods (SABR, interpolation).
    
    Example:
        pinn-compare --model models/best_model.pt --output results/comparison.csv
    """
    logger.info(f"Loading configuration from {config}")
    logger.info(f"Loading PINN model from {model}")
    
    # TODO: Implement baseline comparisons
    logger.info("Baseline comparison functionality coming soon")
    
    return 0


if __name__ == "__main__":
    cli()
