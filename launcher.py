import os
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_CLIENT_SHOW_ERROR_DETAILS'] = 'false'

# Suppress Streamlit's telemetry prompt
os.environ['STREAMLIT_TELEMETRY_OPTOUT'] = 'true'

import subprocess
import sys

# Run streamlit without prompts
subprocess.run([
    sys.executable, '-m', 'streamlit', 'run', 'app.py',
    '--server.port', '8502',
    '--client.showErrorDetails=false'
], env={**os.environ, 'STREAMLIT_SERVER_HEADLESS': 'true'})
