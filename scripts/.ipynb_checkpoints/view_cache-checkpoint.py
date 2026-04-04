import os

# kaggle automatically downloads in a cache location
# used this to verify file structure

path = "/home/user/path/here/.cache/kagglehub/datasets/ravidussilva/real-ai-art/versions/5/Real_AI_SD_LD_Dataset"

if os.path.isdir(path):
    print(f"Directory contents for {path}:")
    for filename in os.listdir(path):
        print(f"- {filename}")
else:
    print(f"Downloaded file: {path}")
    
