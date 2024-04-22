# Create Folder for Background Noise
BACKGROUND_NOISE_DOWNLOAD_PATH=datasets/downloads/background_noise
mkdir -p $BACKGROUND_NOISE_DOWNLOAD_PATH

# Download Background Noise
wget -d https://github.com/karoldvl/ESC-50/archive/master.zip -O ESC-50-master.zip
mv ESC-50-master.zip $BACKGROUND_NOISE_DOWNLOAD_PATH

# Unzip Background Noise
cd $BACKGROUND_NOISE_DOWNLOAD_PATH
unzip ESC-50-master.zip