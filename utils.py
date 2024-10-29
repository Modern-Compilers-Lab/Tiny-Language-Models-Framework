from pathlib import Path
import glob
import os

def extract_neptune_id():
    """
    Extract Neptune ID from a configuration file.
        
    Returns:
        str: Neptune ID if found, None otherwise
    """
    directory = Path(__file__).resolve().parent
    search_pattern = os.path.join(directory, "*.conf")
    conf_files = glob.glob(search_pattern)
    if not conf_files:
        return None
    conf_file  = conf_files[0]

    try:
        with open(conf_file, 'r') as file:
            for line in file:
                if line.startswith('neptune-id:'):
                    # Strip whitespace and split on colon
                    neptune_id = line.split(':', 1)[1].strip()
                    return neptune_id
        return None
    except FileNotFoundError:
        print(f"Error: File '{conf_file}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None