# EfficientMTNet: Brain Tumor Detection Project

This repository contains the **EfficientMTNet** architecture - a novel lightweight deep learning model for simultaneous brain tumor classification and segmentation.

## 📁 Repository Structure

```
.
├── EfficientMTNet/          # Complete model implementation
│   ├── model/              # Architecture code
│   ├── datasets/           # Data loaders
│   ├── losses/             # Loss functions
│   ├── utils/              # Utilities
│   ├── train.py           # Training script
│   ├── test.py            # Testing script
│   ├── demo.py            # Quick demo
│   ├── README.md          # Full documentation
│   └── QUICKSTART.md      # Quick start guide
│
└── brain_tumer_dataset/    # Dataset (MRI scans)
```

## 🚀 Getting Started

All code and documentation is in the **`EfficientMTNet/`** folder:

```bash
cd EfficientMTNet
```

See [EfficientMTNet/README.md](EfficientMTNet/README.md) for complete documentation.

See [EfficientMTNet/QUICKSTART.md](EfficientMTNet/QUICKSTART.md) for quick setup instructions.

## 🎯 Quick Demo

Test the model architecture:

```bash
cd EfficientMTNet
pip install -r requirements.txt
python demo.py
```

## 📊 Model Highlights

- **Parameters:** ~4M (lightweight)
- **Speed:** ~45 FPS
- **Architecture:** Novel multi-task design
- **Features:** Deep supervision, efficient attention, multi-scale fusion

## 📖 Documentation

Complete documentation is available in the [EfficientMTNet](EfficientMTNet/) directory.

## 📄 License

This project is for educational and research purposes.
