import subprocess

def Retrain(script_filenames):
    # Replace 'script_to_run.py' with the actual filename you want to run
    for script_filename in script_filenames:
        try:
            subprocess.run(['python', script_filename], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
    
if __name__ == "__main__":
    scripts_to_run = ['SplitData.py', 'PreprocessData.py', 'Train.py']
    Retrain(scripts_to_run)
