# Create virtual environment
python -m venv ic_env
source ic_env/bin/activate  # On Windows: ic_env\Scripts\activate

# Install dependencies
pip install numpy scipy pandas matplotlib seaborn
pip install biopython networkx scikit-learn
pip install requests python-bitcoinlib
pip install h5py jupyter statsmodels


