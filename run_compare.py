import subprocess, sys, os
project_dir = os.path.expanduser("~/My Stuff/SIH_PS26012")
env = os.environ.copy()
env["PYTHONPATH"] = project_dir + ":" + env.get("PYTHONPATH", "")
env["PYTHONUNBUFFERED"] = "1"
r = subprocess.run(
    [sys.executable, "-u", "-m", "src.eval.compare", "--limit", "150"],
    cwd=project_dir, env=env, capture_output=True, text=True
)
print(r.stdout)
print(r.stderr)
