# PolarDETR:  Enhancing Interpretability in Multi-modal Methods for Jawbone Lesion Detection in CBCT

PolarDETR is a deep learning framework for the precise detection and localization of anatomical entities in dental images based on text descriptions. The system uses a polar coordinate encoding approach combined with anatomical constraints for improved accuracy and interpretability.

## Core Features

- **Polar Text-Position Encoding (PTPE)**: Maps textual descriptions to polar coordinates
- **Anatomical Constraint Learning**: Incorporates domain knowledge about anatomical regions
- **Position Matching Optimization**: Aligns predicted regions with text descriptions
- **Interpretability Module**: Provides anatomical consistency and position-matching scores
- **FDI Tooth Notation Support**: Uses the international FDI tooth numbering system

## Project Structure

```
PolarDETR/
├── configs/                        # Configuration files
├── data/                           # Data preprocessing and loading utilities
│   ├── dataset.py                  # Dataset for main model training
│   └── FDI_MATCH.xml               # FDI tooth notation to angle mapping
├── models/                         # Model architecture components
│   ├── decoders/                   # DETR decoder with position enhancements
│       ├── position_matching.py    # Position Matching Optimization
        ├── anatomy_constraint.py   # Anatomical Constraint Learning
│   └── encoders/                   # Text and image encoders
        ├── ptpe.py                 # Polar Text-Position Encoding
├── utils/                          # Helper functions and evaluation metrics
│   ├── fdi_parser.py               # FDI tooth notation parser
│   ├── metrics.py                  # Evaluation metrics
│   └── visualization.py            # Visualization utilities
├── main.py                         # Main training script
├── inference.py                    # Inference script
├── demo.py                         # Demo script for PTPE visualization
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## Installation

```bash
git clone https://github.com/xxxxxxxx/PolarDETR.git
cd PolarDETR
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --config configs/default.yaml
```

### Inference

```bash
python inference.py --model_path checkpoints/model.pth --image path/to/image.dcm --text "3mm cyst distal to tooth 37"
```

### Demo (do not use BioClincalBERT)

```bash
python demo.py --text "3mm cyst distal to tooth 37" --fdi_xml data/FDI_MATCH.xml
```

## Technical Details

### FDI Tooth Notation

The system uses the FDI (Fédération Dentaire Internationale) tooth numbering system:

- First digit indicates the quadrant (1-4)
- Second digit indicates the tooth position in the quadrant (1-8)
- Example: 37 = lower left second molar

### Polar Text-Position Encoding (PTPE)

The PTPE module converts textual descriptions of anatomical locations into polar coordinates with the following process:

1. **Entity Extraction**: Extract anatomical entities (quadrant, tooth number, distance, direction)
2. **Polar Mapping**: Map entities to polar coordinates (r, θ)
3. **Position Encoding**: Generate a 3D encoding [sin(θ), cos(θ), log(r+1)]

### Evaluation Metrics

- **Anatomical Consistency Score (ACS)**: Measures alignment with anatomical regions
- **Position Matching Score (PMS)**: Measures alignment with text-derived positions
- **Standard Detection Metrics**: mAP

## BioClinicalBERT Fine-tuning for Dental Entity Extraction

Fine-tuning BioClinicalBERT to extract dental entities from text descriptions for the PolarDETR system.

### Entities Extracted

1. **Tooth Number** (FDI notation): Two-digit number (quadrant 1-4, position 1-8)
2. **Distance**: Distance in millimeters
3. **Direction**: Relationship (mesial, distal, buccal, lingual, labial, palatal, apical, coronal)
4. **Quadrant**: Dental quadrant (1-4)

### Data Format

```json
{
  "id": 1,
  "text": "3mm cyst distal to tooth 36",
  "tooth_number": 36,
  "distance": 3.0,
  "direction": 1,
  "quadrant": 3
}
```

- `direction`: 0=mesial, 1=distal, 2=buccal, 3=lingual, 4=labial, 5=palatal, 6=apical, 7=coronal
- `quadrant`: 1=upper right, 2=upper left, 3=lower left, 4=lower right

### Setup

#### Prerequisites

```bash
pip install torch transformers pandas scikit-learn tqdm pyyaml
```

#### Training

```bash
python train_bioclinicalbert.py --config ./configs/config_finetune.yaml
```

## Citation

If you use this code in your research, please cite:

```
@article{PolarDETR2025,
  title={PolarDETR: Anatomical Entity Detection and Localization in Dental Images},
  author={Author, A. and Author, B.},
  journal={xxxxxxxxxx},
  year={2025}
}
```
