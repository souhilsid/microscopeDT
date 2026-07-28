# MicroscopeDT Final Demo

This folder is the clean final version of the project. It packages the final demo app and points the interface to the saved quick attention experiment artifacts for inspection and use.

## Final Model Choice

The Streamlit demo now uses the saved results under:

```text
artifacts_quick_attention_experiments/
```

By default, the app selects the saved image-capable experiment with the lowest half-life MAE so that generated bubble-frame examples are shown immediately. The main image-capable quick attention result is `production_attention`, which keeps the half-life error below 20 seconds in the saved quick evaluation. Other saved models, including the ensemble variants, can be selected from the sidebar dropdown.

## Folder Structure

```text
Final/
  app/
    streamlit_app.py
  data/
    Ready_data.csv
    README_DATA.md
  model/
    best_serious_convlstm.pt
    config.json
    tabular_preprocessor.pkl
  results/
    controlled_comparison_summary.csv
    split_summary.csv
    test_half_life_predictions.csv
    test_metrics.csv
    training_history.csv
  quick_results/              # optional standalone copy of quick artifacts
  src/
    bubble_convlstm_serious.py
  requirements.txt
  README.md
```

The app first looks for quick results in the parent project folder:

```text
artifacts_quick_attention_experiments/
```

If you want the `Final/` folder to run as a standalone package, copy that folder into `Final/quick_results/`.

## Run Locally

From the project root:

```powershell
cd Final
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

## What the App Shows

The Streamlit app includes:

- A model dropdown for the saved quick attention and ensemble experiments.
- Saved test metrics for the selected model.
- A comparison table for all saved quick experiment metrics.
- Half-life prediction traces.
- Side-by-side image comparison:
  - last input frame
  - original target frame
  - generated next frame
  - normalized absolute error
- Actual, predicted, and absolute half-life error for selected image examples.
- Downloadable prediction CSV files.

Generated image examples are available for image-producing models such as `production_attention`, `baseline_original`, and `production_attention_finetuned`. Ensemble and aggregation models are half-life-only results, so the app shows their metrics and prediction tables without generated images.

## Deployment Notes

For a local network demo:

```powershell
cd Final
python -m streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Then open:

```text
http://localhost:8501
```

For deployment to a hosted server:

1. Copy the full `Final/` folder to the server.
2. Copy `artifacts_quick_attention_experiments/` into `Final/quick_results/`.
3. Install Python dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Start Streamlit:

   ```powershell
   python -m streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
   ```

5. Make sure port `8501` is allowed by the firewall or reverse proxy.

For Streamlit Community Cloud, include `Final/quick_results/` with the saved metrics, prediction CSVs, and generated image examples needed for the demo.
