import os, json

class Logger:
    def __init__(self, project_name, run_name = "default", log_dir="logs", force_reset = True):
        self.project_name = project_name
        self.run_name = run_name
        self.force_reset = force_reset
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"{project_name}_{run_name}.jsonl")
        self._initialized_files = set()

    def _ensure_file_reset(self, filepath):
        """Reset file on first write."""
        if filepath not in self._initialized_files:
            with open(filepath, "w") as f:
                pass  # Create/truncate file
            self._initialized_files.add(filepath)
    
    def log(self, t, log):
        if self.force_reset:
            self._ensure_file_reset(self.log_file)
        content = {
            "type": t,
            "log": log
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(content) + "\n")