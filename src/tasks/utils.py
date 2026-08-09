import os
import shutil
import tempfile
import importlib.util

def load_model(model_version: str, exp_dir: str = None):
    """SceneNAT model class loading"""
    if exp_dir and os.path.exists(exp_dir):
        try:
            models_script_dir = os.path.join(exp_dir, "models_script")
            if not os.path.exists(models_script_dir):
                print(f"Models script directory not found: {models_script_dir}, trying to load from source.")
            else:
                target_file = f"scene_nat_{model_version}.py"
                target_path = os.path.join(models_script_dir, target_file)
                
                if os.path.exists(target_path):
                    import sys
                    
                    temp_dir = tempfile.mkdtemp()
                    try:
                        temp_models_script = os.path.join(temp_dir, "models_script")
                        shutil.copytree(models_script_dir, temp_models_script)

                        # Create __init__.py files to make directories packages
                        with open(os.path.join(temp_dir, "__init__.py"), "w") as f:
                            pass
                        with open(os.path.join(temp_models_script, "__init__.py"), "w") as f:
                            pass
                        
                        # Also copy sibling directories like 'networks' if they exist and are needed
                        models_dir = os.path.dirname(os.path.dirname(target_path)) # Parent of models_script
                        for item in os.listdir(models_dir):
                            s = os.path.join(models_dir, item)
                            d = os.path.join(temp_models_script, item)
                            if os.path.isdir(s) and item != 'models_script' and not os.path.exists(d):
                                shutil.copytree(s, d)

                        sys.path.insert(0, temp_dir)
                        
                        module_name = f"models_script.scene_nat_{model_version}"
                        module = importlib.import_module(module_name)

                    finally:
                        if temp_dir in sys.path:
                            sys.path.remove(temp_dir)
                        shutil.rmtree(temp_dir)
                    
                    class_name = f"SceneNAT_{model_version}"
                    if hasattr(module, class_name):
                        model = getattr(module, class_name)
                        print(f"Successfully loaded SceneNAT class: {class_name} from resumed experiment")
                        return model
                    else:
                        print(f"Class {class_name} not found in {target_file}")
                else:
                    print(f"Model version {model_version} not found in models_script directory, trying to load from source.")
        except Exception as e:
            print(f"Error loading SceneNAT class for version {model_version} from experiment: {e}")

    try:
        class_name = f"SceneNAT_{model_version}"
        module = __import__(f"src.models", fromlist=[class_name])
        model_class = getattr(module, class_name)
        print(f"Successfully loaded SceneNAT class: {class_name} from source")
        return model_class
    except (ImportError, AttributeError) as e:
        print(f"Error importing SceneNAT class for version {model_version} from source: {e}")
        return None
