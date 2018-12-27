# RTBT: Real-time beat tracking python module for Raspberry Pi
### This python module was built as part of a submission for the *IEEE Signal Processing Cup 2017*

Please read the [README.pdf](https://github.com/AhmedImtiazPrio/RTBT/blob/master/README.pdf) for details on running the code and the following [paper](https://ieeexplore.ieee.org/abstract/document/8397024) for details on the algorithm.

Al-Hussaini, I., Humayun, A.I., Alam, S., Foysal, S.I., Al Masud, A., Mahmud, A., Chowdhury, R.I., Ibtehaz, N., Zaman, S.U., Hyder, R. and Chowdhury, S.S., 2018, April. Predictive Real-Time Beat Tracking from Music for Embedded Application. In 2018 *IEEE Conference on Multimedia Information Processing and Retrieval* (MIPR) (pp. 297-300). IEEE.

#### The algorithm and embedded application had won an *Honorable Mention* in the *IEEE Signal Processing Cup 2017*. Watch the demo video [here](https://youtu.be/fyENs0ABZhw).

### Dependencies:

- Python (2.7)
- Numpy
- Scipy
- ffmpeg
- PyAudio
- CFFI (C Foreign Function Interface for Python)
- Six (Python 2 and Python 3 compatibility library)
- PortAudio Version 19
- serial (only required if paired with an Arduino)

### Basic Usage:

```

from BeatTracker import BeatTracker
proc = BeatTracker()
proc.Beats(InFile=’sample.wav’, OutFile=’sample.txt’)

```
Read the README.pdf for details on the BeatTracker Class and its submodules


### Cite:
```
@inproceedings{al2018predictive,
  title={Predictive Real-Time Beat Tracking from Music for Embedded Application},
  author={Al-Hussaini, Irfan and Humayun, Ahmed Imtiaz and Alam, Samiul and Foysal, Shariful Islam and Al Masud, Abdullah and Mahmud, Arafat and Chowdhury, Rakibul Islam and Ibtehaz, Nabil and Zaman, Sums Uz and Hyder, Rakib and others},
  booktitle={2018 IEEE Conference on Multimedia Information Processing and Retrieval (MIPR)},
  pages={297--300},
  year={2018},
  organization={IEEE}
}

```
