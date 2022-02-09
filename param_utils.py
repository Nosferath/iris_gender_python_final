import json
from pathlib import Path


def generate_vgg_full_params(
        epochs: int,
        use_val: bool,
        learning_rate: float,
        batch_size: int,
        architecture: str,
        out_file
):
    """Generates a .JSON file with the paramters for full-vgg tests.
    Other parameters such as dataset index, partition, and use_peri
    should be set using the CLI options.
    """
    params = {
        'epochs': epochs,
        'use_val': use_val,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'architecture': architecture
    }
    out_file = Path(out_file)
    out_file.parent.mkdir(exist_ok=True, parents=True)
    with open(out_file, 'w') as f:
        json.dump(params, f, indent=4)



