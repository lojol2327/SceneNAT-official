# conda create -n nat --file packages.txt
conda activate nat

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt

python -c "import nltk; nltk.download('cmudict')"

python -m pip install openai-clip
python -m pip install matplotlib