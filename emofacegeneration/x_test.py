import audio
from os.path import join
import matplotlib.pyplot as plt
import numpy as np

syncnet_T = 5
syncnet_mel_step_size = 16

from models import Wav2Lip as Wav2Lip

def crop_audio_window(spec, start_frame):
    if type(start_frame) == int:
        start_frame_num = start_frame
    # else:
    #     start_frame_num = self.get_frame_id(start_frame) # 0-indexing ---> 1-indexing
    start_idx = int(80. * (start_frame_num / float(25)))
    
    end_idx = start_idx + syncnet_mel_step_size

    return spec[start_idx : end_idx, :]


def get_segmented_mels(spec, start_frame):
    mels = []
    assert syncnet_T == 5
    start_frame_num = start_frame + 1 # 0-indexing ---> 1-indexing

    if start_frame_num - 2 < 0: return None
    for i in range(start_frame_num, start_frame_num + syncnet_T):
        m = crop_audio_window(spec, i - 2)
        if m.shape[0] != syncnet_mel_step_size:
            return None
        mels.append(m.T)

    mels = np.asarray(mels)
    return mels


def main():
    vidname = "/mnt/DataSets/tish386/wav2lip_preprocess_chunk_1_archive/_0VYHRpcjIw/00193"
    wavpath = join(vidname, "audio.wav")
    wav = audio.load_wav(wavpath, 16000)

    orig_mel = audio.melspectrogram(wav).T
    mel = crop_audio_window(orig_mel.copy(), 10)
    
    print(orig_mel.shape)
    print(mel.shape)

    plt.imshow(orig_mel)
    plt.colorbar()
    plt.show()
    # plt.figure()
    # fig, axarr = plt.subplots(2)
    # axarr[0].imshow(orig_mel)
    # axarr[1].imshow(mel)
    
    # plt.colorbar()
    # plt.show()

    indiv_mels = get_segmented_mels(orig_mel.copy(), 10)
    for i in indiv_mels:
        print(i.shape)
    # print(len(indiv_mels))
    # for i in indiv_mels:
    #     plt.imshow(i)
    #     plt.colorbar()
    #     plt.show()


def main_V2():
    device = 'cpu'
    model = Wav2Lip().to(device)

    print(Wav2Lip)

if __name__ == "__main__":
    main_V2()