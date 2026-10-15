Comparative Study of Traditional Machine Learning and Transformer Models for Spam Classification
This project compares three classifiers for email Spam/Ham detection:
1. TF-IDF + Logistic Regression
2. TF-IDF + Multinomial Naive Bayes
3. Fine-tuned DistilBERT using a custom PyTorch training loop (no Hugging Face Trainer API)
It was organized around the requirements in the supplied project assignment: data exploration, a stratified split, Accuracy/Precision/Recall/F1, confusion matrices, metric comparison, transformer training behaviour, training-time comparison, and a report template.
1. Get the dataset
Download the dataset ZIP from:
https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification
Extract the ZIP, find the CSV file, and copy the CSV into data/raw/. For example:
data/
└── raw/
    ├── README.md
    └── spam_email.csv   <- your downloaded CSV goes here
The CSV filename can be different. The code attempts to infer common text and label column names. If it chooses the wrong file or column, use --data-path and adjust infer_columns() in src/common.py.
Check label meaning: the code assumes numeric labels 0 = Ham and 1 = Spam. If your CSV uses numeric labels with the opposite meaning, update normalise_label() in src/common.py before training. Text labels such as spam and ham are handled automatically.
2. Install on Windows
Python 3.10 or 3.11 is recommended for a straightforward setup. Open this folder in VS Code, then choose Terminal → New Terminal.
Run these commands one at a time in PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
If PowerShell blocks activation for this terminal, run this once in the same terminal and activate again:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
If python is not recognized, install Python from https://www.python.org/downloads/ and check Add Python to PATH during setup.
3. Explore the dataset first
For the first run, use a stratified sample of up to 12,000 cleaned examples. This keeps the first experiment more manageable on a normal computer.
python src\explore_data.py --max-samples 12000
Check the printed output, results/data_profile.json, and results/class_distribution.png. Confirm that the detected label column and label meanings are correct.
Use --max-samples 0 to run on all cleaned rows. For a fair model comparison, pass the same --max-samples setting to every script.
4. Train the two traditional models
python src\train_ml.py --max-samples 12000
This trains Logistic Regression and Naive Bayes with the same TF-IDF settings. It saves the fitted pipelines under artifacts/ and metrics, predictions, and confusion matrices under results/.
5. Fine-tune DistilBERT with PyTorch
python src\train_distilbert.py --max-samples 12000 --epochs 2
The first run downloads pretrained DistilBERT files and therefore needs an internet connection. Fine-tuning can take a long time on CPU; a GPU runtime such as Google Colab is recommended if your computer does not have a compatible GPU. Never describe a run as GPU-based unless the script prints a CUDA device.
Default choices:
- Model: distilbert/distilbert-base-uncased
- Learning rate: 2e-5
- Batch size: 8
- Maximum token length: 128
- Epochs: 2
- Optimizer: AdamW with weight decay 0.01
- Linear learning-rate schedule with warmup
- Gradient clipping: 1.0
- Layer freezing: none by default; all model layers can be fine-tuned
- Train/validation/test: 70%/10%/20%, stratified; the 20% test partition uses the same seed and split as the classical experiments
Optional settings:
python src\train_distilbert.py --max-samples 12000 --epochs 3 --batch-size 8 --max-length 128 --learning-rate 0.00002
For a smaller faster pilot, you can use --max-samples 2000. For a final full-dataset run, use --max-samples 0 for all scripts and record the device and exact settings in your report.
6. Create comparison plots
After both model scripts complete, run:
python src\plot_results.py
Expected outputs include:
- results/class_distribution.png
- results/metrics_comparison.png
- results/confusion_matrix_logistic_regression.png
- results/confusion_matrix_naive_bayes.png
- results/confusion_matrix_distilbert.png
- results/transformer_training_curve.png
- results/training_time_comparison.png
- results/all_model_metrics.csv
Each training script creates its own confusion matrix. If DistilBERT has not completed, the comparison plot will show only the available classical models.
7. Write the technical report
Open report/technical_report_template.md. It follows the required report sections and tells you which plot to insert. Replace every bracketed placeholder with your actual values from the files in results/. Do not invent results. Discuss false positives and false negatives, not just accuracy.
The template includes the main limitations to consider: sample size, maximum token length, hardware-dependent training times, label-column inference, and the fact that one train/test split does not measure variability across multiple runs.
Project structure
Spam_Classification_Project/
├── data/raw/                 # Put the Kaggle CSV here; excluded from Git
├── artifacts/                # Saved trained models; excluded from Git
├── results/                  # Metrics, plots, predictions; excluded from Git
├── report/
│   └── technical_report_template.md
├── src/
│   ├── common.py
│   ├── explore_data.py
│   ├── train_ml.py
│   ├── train_distilbert.py
│   └── plot_results.py
├── requirements.txt
└── README.md
Method choices in brief
- Stratified split: preserves approximate Ham/Spam proportions in the splits.
- TF-IDF unigrams + bigrams: represents individual terms and short phrases while limiting the vocabulary to 100,000 features.
- Light cleaning: strips HTML tags and substitutes URL/email tokens; it does not remove all punctuation or stop words because such patterns can help identify spam.
- Consistent evaluation: spam is class 1 and the positive class for precision, recall, and F1; confusion matrices show Ham and Spam explicitly.
- Compute-time caveat: traditional model fit time includes fitting TF-IDF; DistilBERT fine-tuning time excludes model-download and tokenization/setup time, which is saved separately in results/transformer_run_config.json.
References
- Dataset: https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification
- Hugging Face DistilBERT documentation: https://huggingface.co/docs/transformers/en/model_doc/distilbert
- Scikit-learn TF-IDF documentation: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
