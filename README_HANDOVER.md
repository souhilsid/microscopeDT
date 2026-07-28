# Microscopy Full Project Handover

This folder is the clean handover package for the microscopy bubble forecasting and foam half-life project.

## Folder Structure

- `01_data_preparation_initial_ml/`
  - Data extraction, initial EDA, and initial ML pipeline materials.
  - Main notebook: `ml_pipeline.ipynb`.
  - Additional extraction reference: `Untitled-1_extraction_reference.ipynb`.
  - Includes EDA plots, ML result outputs, model plots, and prepared tabular outputs.

- `02_current_modeling_workspace/`
  - Current full modeling project workspace.
  - Includes ConvLSTM scripts, controlled comparison scripts, quick attention experiments, reports, training summaries, and artifacts.
  - The large image pickle is not duplicated here; use `shared_data/Ready_images.pkl`.

- `app_demo/`
  - Streamlit demo app, copied from the final app folder.
  - Contains the app interface, packaged model files, app results, and deployment/demo utilities.
  - The app can use `shared_data/Ready_images.pkl` from this handover folder.

- `shared_data/`
  - Full ready data used by the final modeling/app workflow.
  - Contains `Ready_data.csv` and `Ready_images.pkl`.

## Raw Data

The original raw/clean folder `Clean Data for Dr. Fadwa` is not copied into this handover package because the team already has it. If extraction needs to be rerun, use the team's existing copy of that folder.

## Notes

- The full image pickle is intentionally stored once in `shared_data/` to avoid duplicate 2.48 GB copies.
- `app_demo/data/Ready_images.pkl` may be absent or may contain a smaller demo file; the app has been updated to also look for `../shared_data/Ready_images.pkl`.
- For online deployment, use the smaller deployment files or a storage bucket for the full `Ready_images.pkl`.
