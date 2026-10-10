# Start the RoboTrace demo with the demo annotation settings (this PowerShell session only; .env is not changed).
#   Model keyframes every 0.5 s (2 fps), split into overlapping 12 s windows sent 4 at a time;
#   every frame analysed locally at the video's native frame rate to tighten the first/last boundaries.
#   Usage (from the project folder):  powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1
$env:ROBOTRACE_SAMPLING = "windowed"
$env:ROBOTRACE_FRAME_SPACING_S = "0.5"
$env:ROBOTRACE_WINDOW_OVERLAP_S = "1.0"
$env:ROBOTRACE_MAX_FRAMES = "24"
$env:ROBOTRACE_MAX_WINDOWS = "8"
$env:ROBOTRACE_PARALLEL_REQUESTS = "4"
$env:ROBOTRACE_MOTION_TRIM = "1"
$port = if ($args.Count -gt 0) { $args[0] } else { "8502" }
& "$PSScriptRoot\..\.venv\Scripts\python.exe" -m streamlit run "$PSScriptRoot\..\app.py" --server.port $port
