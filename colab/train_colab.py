"""Run this in Google Colab to train and export model artifacts."""

from pathlib import Path
import shutil

from google.colab import drive


def main() -> None:
    drive.mount('/content/drive')

    project_dir = Path('/content/Derivative-Hedging')
    if not project_dir.exists():
        raise FileNotFoundError('Upload or git clone the project to /content/Derivative-Hedging first.')

    # After installing requirements in a notebook cell:
    # !pip install -r /content/Derivative-Hedging/requirements.txt
    # !python /content/Derivative-Hedging/scripts/prepare_lse_data.py
    # !python /content/Derivative-Hedging/train_lse.py

    artifact = project_dir / 'artifacts' / 'recurrent_ppo_lse.zip'
    if not artifact.exists():
        raise FileNotFoundError(f'Model artifact not found: {artifact}')

    export_dir = Path('/content/drive/MyDrive/derivative_hedging_exports')
    export_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact, export_dir / artifact.name)
    print(f'Exported model to {export_dir / artifact.name}')


if __name__ == '__main__':
    main()
