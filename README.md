# sprite-creator
A Python tool for creating voice-synced toonified sprites from video clips

## Getting Started
This tool provides an automated workflow for creating animated recordings of continuous human voice. The generated files can be easily utilized as sprites in Unity for developing game prototypes for research in auditory and speech perception.

### Installation and Usage
- Clone the repository:
``` 
git clone https://github.com/pr0m1th3as/sprite-creator.git
cd sprite-creator
```
- Run main script:
```
python3 createsprite.py
```
### Default settings
By default, `createsprite.py` batch processes all video clips with `.mp4` extension available in `/videos` assuming they are recorded at 25fps. Audio channels are merged to monophonic channel, the first 0.4 seconds of the video are ignored for avoiding clipping or other sound artifacts and processed voice trimmed audio is stored in `/voice`. Every 5th frame from the video clip is extracted, starting about 300ms before the voice signal and ending about another 300ms after it stops. Extracted frames are stored in `/frames`, whereas the toonified version of the face appearing in these frames is stored in `the /toons` folder. Toonified face images are limited to 256x256 pixels in size. Shortened video clips mimmicking the targeted sprites with the clipped monophonic audio synced at 5 fps are stored in the `/sprites` folder. Most of these parameters are hardcoded at the beginning of the `createsprite.py` script and can be easily modified to meet your requirements. There are 5 different cartoon styles to choose from located in `/saved_models`. `createsprite.py` uses style 3, by default.

### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN
- Python 3

### Python dependencies:
- ffmpy=0.3.0
- librosa=0.9.2
- moviepy=1.0.3
- noisereduce=2.0.1
- numpy=1.23.0
- opencv-python=4.5.3.56
- opencv-python-headless=4.5.3.56
- pandas=1.5.3
- scikit-image=0.18.3
- torch=1.12.1

### Attribution
Pretrained models and certain code has been forked/adapted from MingtaoGuo's [CartoonBANK](https://github.com/MingtaoGuo/CartoonBANK) repository.
