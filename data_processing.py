from imports import *


Ravdess = "coursework_datasets/Ravdess/audio_speech_actors_01-24/"
Tess = "coursework_datasets/Tess/TESS Toronto emotional speech set data/TESS Toronto emotional speech set data/"
Crema = "coursework_datasets/Crema/AudioWAV"
Savee = "coursework_datasets/Savee/AudioData/AudioData/"

crema_label_dict = {'NEU': 'neutral', 'HAP': 'happy', 'SAD': 'sad', 'ANG': 'angry', 'FEA': 'fear', 'DIS': 'disgust'}
tess_label_dict = {'ps': 'surprise', 'neutral': 'neutral', 'happy': 'happy', 'sad': 'sad', 'angry': 'angry','fear': 'fear', 'disgust': 'disgust'}
savee_label_dict = {'su': 'surprise', 'n': 'neutral', 'h': 'happy', 'sa': 'sad', 'a': 'angry','f': 'fear', 'd': 'disgust'}
def prepare_dataset(dataset_path, dataset_label_dict):
    directory = os.listdir(dataset_path)
    emotion_file = []
    path_to_emotion_file = []
    if dataset_path == Crema:
        for file in directory:
            split_file_name = file.split(".")[0]
            split_file_name = file.split('_')[2]
            if split_file_name in dataset_label_dict:
                emotion_file.append(dataset_label_dict[split_file_name])
                path_to_emotion_file.append(dataset_path + '/' + file)
            else:
                emotion_file.append('unfamiliar')
                path_to_emotion_file.append(Crema + '/' + file)
    elif dataset_path == Ravdess:
        for dct in directory:
            actor_numb = os.listdir(dataset_path + dct)
            for file in actor_numb:
                split_file_name = file.split('.')[0]
                split_file_name = split_file_name.split('-')
                emotion_file.append(int(split_file_name[2]))
                path_to_emotion_file.append(dataset_path + dct + '/' + file)
    elif dataset_path == Tess:
        for dct in directory:
            emotion_folder = os.listdir(dataset_path + dct)
            for folder in emotion_folder:
                split_file_name = folder.split('.')[0]
                split_file_name = split_file_name.split('_')[2]
                emotion_file.append(dataset_label_dict[split_file_name])
                path_to_emotion_file.append(dataset_path + dct + '/' + folder)
    elif dataset_path == Savee:
        for dct in directory:
            actor_folder = os.listdir(dataset_path + dct)
            for folder in actor_folder:
                letter = str(folder[0])
                sec_letter = str(folder[1])
                if (letter == 's' and sec_letter == 'a'):
                    emotion_file.append(dataset_label_dict['sa'])
                    path_to_emotion_file.append(dataset_path + dct + '/' + folder)
                elif (letter == 's' and sec_letter == 'u'):
                    emotion_file.append(dataset_label_dict['su'])
                    path_to_emotion_file.append(dataset_path + dct + '/' + folder)
                else:
                    emotion_file.append(dataset_label_dict[letter])
                    path_to_emotion_file.append(dataset_path + dct + '/' + folder)
    emotion_df = pd.DataFrame(emotion_file, columns=['Emotions'])
    path_df = pd.DataFrame(path_to_emotion_file, columns=['Path'])
    df = pd.concat([emotion_df, path_df], axis=1)
    return df


def data_augmentation(aug_type, audio, sr):
    if aug_type == 'noise':
        noise_factor = 0.005
        noise = np.random.randn(len(audio))
        augmented_audio = audio + noise_factor * noise
        return augmented_audio
    elif aug_type == 'time_stretch':
        min_rate = 0.7
        max_rate = 1.3
        rate = np.random.uniform(min_rate, max_rate)
        stretched_audio = librosa.effects.time_stretch(audio, rate=rate)
        return stretched_audio
    elif aug_type == 'pitch_shift':
        min_semitones = -4
        max_semitones = 4
        n_steps = np.random.randint(min_semitones, max_semitones)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def extract_features(data, sr):
    features = np.array([])

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    features = np.hstack((features, chroma_stft))

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13).T, axis=0)
    features = np.hstack((features, mfcc))

    # Mel Spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
    features = np.hstack((features, mel))

    # Contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
    features = np.hstack((features, contrast))

    # Tonnetz
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sr).T, axis=0)
    features = np.hstack((features, tonnetz))

    # Prosodic Features
    pitch, _ = librosa.piptrack(y=data, sr=sr)
    pitch_mean = np.mean(pitch)
    pitch_std = np.std(pitch)
    features = np.hstack((features, pitch_mean, pitch_std))

    #Speaking Rate
    speaking_rate = librosa.onset.onset_detect(y=data, sr=sr)
    speaking_rate = len(speaking_rate) / (data.shape[0] / sr)
    features = np.hstack((features, speaking_rate))

    return features


def get_features(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res_usual = extract_features(data, sample_rate)
    total_res = np.array(res_usual)

    # data with noise
    noise_data = data_augmentation('noise', audio=data, sr=sample_rate)
    res_noise = extract_features(noise_data, sample_rate)
    total_res = np.vstack((total_res, res_noise))

    # data with stretching and pitching
    time_stretch_data = data_augmentation('time_stretch', audio=data, sr=sample_rate)
    stretch_pitch_data = data_augmentation('pitch_shift', audio=time_stretch_data, sr=sample_rate)
    res_stretch_pith = extract_features(stretch_pitch_data, sample_rate)
    total_res = np.vstack((total_res, res_stretch_pith))

    return total_res


def extract_features_single_file(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    features = extract_features(data, sample_rate)
    return features


def analyze_sentiment(audio_file_path):
    audio, sr = librosa.load(audio_file_path)
    pitch, _ = librosa.piptrack(y=audio, sr=sr)
    speaking_rate = librosa.onset.onset_detect(y=audio, sr=sr)
    speaking_rate = len(speaking_rate) / (audio.shape[0] / sr)

    sentiment = "neutral"
    intensity = "moderate"  # Default intensity

    mean_pitch = np.mean(pitch)

    # Tone analysis (pitch)
    if mean_pitch > 170:
        sentiment = "positive"
        intensity = "strong"
    elif mean_pitch > 130:
        sentiment = "positive"
        intensity = "moderate"
    elif mean_pitch < 80:
        sentiment = "negative"
        intensity = "strong"
    elif mean_pitch < 120:
        sentiment = "negative"
        intensity = "moderate"

    # Speed analysis (speaking rate)
    if speaking_rate > 1.8:
        sentiment = "negative"
        intensity = "strong"
    elif speaking_rate > 1.3:
        sentiment = "negative"
        intensity = "moderate"
    elif speaking_rate < 0.8:
        sentiment = "positive"
        intensity = "strong"
    elif speaking_rate < 1.2:
        sentiment = "positive"
        intensity = "moderate"

    # Intonation analysis (pitch variation)
    pitch_std = np.std(pitch)
    if pitch_std > 25:
        sentiment = "negative"
        intensity = "strong"
    elif pitch_std > 15:
        sentiment = "negative"
        intensity = "moderate"
    elif pitch_std < 7:
        sentiment = "positive"
        intensity = "strong"
    elif pitch_std < 13:
        sentiment = "positive"
        intensity = "moderate"

    return sentiment, intensity
