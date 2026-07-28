from pathlib import Path
import runpy


APP_PATH = Path(__file__).resolve().parent / "app_demo" / "app" / "streamlit_app.py"
runpy.run_path(str(APP_PATH), run_name="__main__")
