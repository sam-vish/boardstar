import os

# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
database_path = os.path.join(project_root, "database")

def get_classes():
    return [d for d in os.listdir(database_path) if os.path.isdir(os.path.join(database_path, d))]

def get_subjects(class_name):
    class_path = os.path.join(database_path, class_name)
    return [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]

def get_chapters(class_name, subject):
    subject_path = os.path.join(database_path, class_name, subject)
    return [f.replace('.md', '') for f in os.listdir(subject_path) if f.endswith('.md')]