# AI Postgraduate Portfolio

> **Author:** Augusto Stahlschmidt<br />
> **University:** Universidade do Vale do Rio dos Sinos (UNISINOS)<br />
> **Program:** Pós-graduação em Inteligência Artificial Aplicada<br />

[About](#about) · [Key Skills](#key-skills) · [Featured Projects](#featured-projects) · [Portfolio](#portfolio) · [Setup](#setup) · [License](#license)

## About

A portfolio of end-to-end machine learning and AI projects covering the full lifecycle: data analysis, modeling, evaluation, and system integration.

Each notebook follows a structured methodology: problem statement, EDA, preprocessing, model training, quantitative evaluation, and conclusions. Topics span the full ML spectrum: from classical neural networks built from scratch to deep learning, generative AI, and real-time IoT pipelines.

All projects were developed as part of the **Pós-graduação em Inteligência Artificial Aplicada** program at **Universidade do Vale do Rio dos Sinos** (UNISINOS) during 2025.

## Key Skills

| Area | Strategies & Techniques | Libraries & Tools |
|------|------------------------|-------------------|
| **Environment** | Python · Reproducibility · Jupyter Notebooks | `Python` · `PyTorch` · `numpy` · `pandas` · `scikit-learn` · `matplotlib` · `Jupyter` |
| **Data Processing & EDA** | EDA · Feature engineering · Outlier detection · Encoding & scaling · Temporal feature extraction | `seaborn` |
| **Visualization** | Statistical analysis · PCA visualization · Heatmaps · Learning curves · Confusion matrices | `seaborn` · `plotly` |
| **Machine Learning** | Feature engineering · Class imbalance handling · Cross-validation · Hyperparameter tuning | `imbalanced-learn` ·  `scipy` |
| **Neural Networks** | MLP from scratch · Backpropagation · SGD · Momentum optimizer · BCE / MSE loss | `torch.nn` |
| **Unsupervised Learning** | K-Means · DBSCAN · GMM · RFM analysis · PCA · UMAP · Silhouette scoring | `umap-learn` |
| **Deep Learning** | CNNs · Transfer learning · Fine-tuning · U-Net · ResNet-50 · Batch normalization · Dropout · Data augmentation · Learning rate scheduling | `torchvision` |
| **Reinforcement Learning** | Custom MDP design · Reward shaping · Policy gradient (PPO, A2C) · Value-based learning (DQN) · Experience replay · Hyperparameter optimization | `gymnasium` · `stable-baselines3` · `optuna` |
| **Natural Language Processing** | Sentence embeddings · Semantic similarity · TF-IDF · Error analysis | `sentence-transformers` · `HuggingFace` |
| **Generative AI** | RAG · Dense vector indexing · Few-shot prompting · LLM confidence calibration | `FAISS` · `sentence-transformers` · `Claude API` |
| **Computer Vision** | Semantic segmentation · Image classification · Transfer learning · DICOM preprocessing | `torchvision` · `PIL` |
| **AI Integration** | Real-time ML inference · REST API serving · Model serialization · Time-series feature engineering | `FastAPI` · `joblib` |

## Featured Projects

### Drone Delivery Route Optimization<br>
Custom reinforcement learning environment simulating a drone delivering packages in a grid-based neighborhood. Multiple deep RL algorithms (PPO, DQN, A2C) were trained and compared, with hyperparameter optimization performed using Optuna. The project demonstrates environment design, reward shaping, and algorithm benchmarking.

### Pet Foreground Segmentation<br>
Binary semantic segmentation of pets using a U-Net architecture implemented in PyTorch. The model isolates pet pixels from background using the Oxford-IIIT Pet dataset, trained with a combined BCE + Dice loss to improve segmentation stability and boundary accuracy.

### Story Point Estimation via RAG<br>
Implementation of a Retrieval-Augmented Generation (RAG) pipeline to estimate Agile story points from historical user stories. Dense sentence embeddings are indexed with FAISS for semantic retrieval, and retrieved examples are incorporated into structured prompts for an LLM to produce estimates with explanations.

### Smart Parking Monitoring System<br>
End-to-end IoT + machine learning pipeline for real-time parking occupancy detection. Sensor readings are streamed via MQTT to a FastAPI service that performs ML inference and broadcasts state changes to a live dashboard using Server-Sent Events. The system integrates data ingestion, feature engineering, model deployment, and real-time visualization in a complete ML system architecture.

## Portfolio

| # | Project | Domain | Highlight |
|---|---------|------|-------------------|
| 1 | [Video Game Sales EDA](portfolio/data-visualization/video_game_sales_visualization.ipynb) | Data Visualization | Generated multiple data visualizations using frameworks to extract business insights from sales data |
| 2 | [Banknote Authentication](portfolio/neural-networks/banknote_authentication.ipynb) | Neural Networks | Implemented Perceptron from scratch for a Binary Classification problem |
| 3 | [Breast Cancer Classification](portfolio/neural-networks/breast_cancer_classification.ipynb) | Neural Networks | Implemented Multilayer Perceptron and Backpropagation from scratch for a Binary Classification problem |
| 4 | [Letter Recognition](portfolio/neural-networks/letter_recognition.ipynb) | Neural Networks | Implemented Multilayer Perceptron and Backpropagation from scratch for a Multi-Class Classification problem |
| 5 | [Penguin Species Classification](portfolio/neural-networks/penguin_species_classification.ipynb) | Neural Networks | Experimented with multiple hyperparameter configurations to obtain stable model training across multiple random seeds |
| 6 | [Forest Cover Type Classification](portfolio/neural-networks/forest_covertype_classification.ipynb) | Neural Networks | Compared imbalanced dataset training performance with oversampling and undersampling strategies on a highly imbalanced dataset |
| 7 | [House Price Prediction](portfolio/supervised-learning/house_price_prediction.ipynb) | Supervised Learning | Explored and compared multiple feature engineering strategies on a custom dataset |
| 8 | [Customer Segmentation](portfolio/unsupervised-learning/customer_segmentation.ipynb) | Unsupervised Learning | Built RFM pipeline to identify classical customer archetypes through behavioral clusters |
| 9 | [Firewall Log Clustering](portfolio/unsupervised-learning/firewall_log_clustering.ipynb) | Unsupervised Learning | Engineered log features and text embeddings to discover dominant network behavior clusters |
| 10 | [Drone Delivery Route Optimization](portfolio/reinforcement-learning/drone_delivery_route_optimization.ipynb) | Reinforcement Learning | Benchmarked multiple algorithms on a custom RL environment and improved baseline performance through hyperparameter optimization |
| 11 | [Chest X-Ray Classification](portfolio/deep-neural-networks/chest_xray_classification.ipynb) | Deep Neural Networks | Fine-tuned a ResNet-50 model on DICOM files to correctly identify Pneumonia radiographs |
| 12 | [Pet Foreground Segmentation](portfolio/deep-neural-networks/pet_foreground_segmentation.ipynb) | Deep Neural Networks | Implemented U-Net from scratch with millions of parameters to segment dog and cat images from the picture background |
| 13 | [Fake News Detection](portfolio/natural-language-processing/fake_news_detection.ipynb) | Natural Language Processing | Developed a transformer embedding + logistic regression pipeline to identify fake news misclassification patterns |
| 14 | [Story Point Estimation via RAG](portfolio/generative-ai/story_point_estimation_with_rag.ipynb) | Generative AI | Developed a RAG pipeline using similarity search integrated with a large language model, outperforming TF-IDF + KNN strategies |
| 15 | [Parking Monitoring System](portfolio/iot-and-big-data/parking_monitoring_system_simulation.ipynb) | IoT + Big Data | Implemented a complete real-time IoT + ML pipeline with API integration to monitor a parking system in a dashboard |

## Setup

**Clone the repository**
```bash
git clone <repository-url>
cd portfolio
```

**Create a virtual environment and install dependencies**
```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

**Launch Jupyter**
```bash
jupyter notebook
```

## License

This project is licensed under the [MIT License](portfolio/LICENSE).

All public datasets used are available under their respective licenses.
