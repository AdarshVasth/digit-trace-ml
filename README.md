# Digit Trace

Draw a handwritten digit on an 8 × 8 grid and see which examples from the training data voted for it. Digit Trace uses scikit-learn's k-nearest neighbors (k-NN) classifier and runs on your computer without an API key or paid service.

## Run locally

Use Python 3.12. From the cloned repository, create and activate a virtual environment.

```sh
python -m venv .venv
# macOS or Linux
source .venv/bin/activate
# Windows PowerShell (use this instead)
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:8000**. Select **Try sample** for a hand-designed 7, or use **Clear**, draw your own digit, and select **Classify**. Clicking a cell toggles it. Start a drag on an empty cell to draw or a filled cell to erase. Keyboard users can focus a cell and toggle it with Space or Enter.

Run the tests or print the reproducible evaluation:

```sh
python -m unittest -v
python model.py
```

## How it works

1. **Training** · `model.py` loads scikit-learn's bundled Digits dataset and fits k-NN on a fixed, stratified 75% split. The remaining 25% is used to measure accuracy.
2. **Inference** · `app.js` sends 64 row-major pixel intensities, each from 0 to 16, to `POST /api/predict`. `app.py` validates the input and calls the model on the local server.
3. **Explanation** · The model returns the five nearest training images, their Euclidean distances, and their vote counts. The digit with the most votes wins. Equal votes go to the lowest digit. `GET /api/metrics` exposes the measured holdout result.

The verified run used 1,347 training images and 450 held-out images, with **443 / 450 correct (98.4%)**. This is a result on the bundled images. Freehand browser drawings are binary pixels and can differ substantially from the grayscale training set. A vote fraction is not a calibrated confidence estimate. The model can guess on unfamiliar marks and only rejects completely blank grids. It recognizes one digit at a time. Python's built-in HTTP server is intended for local demos.

## Data

`sample.json` contains an original, hand-designed grid of the digit 7, released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). The repository does not bundle a copy of the training dataset. The [scikit-learn Digits loader](https://scikit-learn.org/1.8/modules/generated/sklearn.datasets.load_digits.html) loads its packaged copy of the [UCI Optical Recognition of Handwritten Digits data](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).

## Internship learning guide

- Start with `DigitModel.explain`. Explain why one 8 × 8 image is represented by 64 features.
- Trace the sample from the browser through the prediction endpoint. Show why 7 wins even though only two neighbors vote for it.
- Draw both a centered digit and an off-center version. Compare predictions and discuss why validation accuracy alone cannot tell you whether a UI works with new inputs.
- As an experiment, compare three and five neighbors with a separate validation split. Keep the test set untouched until you have selected a configuration.

API references: [KNeighborsClassifier](https://scikit-learn.org/1.8/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) · [train_test_split](https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.train_test_split.html) · [Python HTTP server](https://docs.python.org/3.12/library/http.server.html).
