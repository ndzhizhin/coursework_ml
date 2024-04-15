from imports import *

Ravdess = "coursework_datasets/Ravdess/audio_speech_actors_01-24/"
Tess = "coursework_datasets/Tess/TESS Toronto emotional speech set data/TESS Toronto emotional speech set data/"
Crema = "coursework_datasets/Crema/AudioWAV"
Savee = "coursework_datasets/Savee/AudioData/AudioData/"

crema_label_dict = {'NEU': 'neutral', 'HAP': 'happy', 'SAD': 'sad', 'ANG': 'angry', 'FEA': 'fear', 'DIS': 'disgust'}
tess_label_dict = {'ps': 'surprise', 'neutral': 'neutral', 'happy': 'happy', 'sad': 'sad', 'angry': 'angry','fear': 'fear', 'disgust': 'disgust'}
savee_label_dict = {'su': 'surprise', 'n': 'neutral', 'h': 'happy', 'sa': 'sad', 'a': 'angry','f': 'fear', 'd': 'disgust'}



def predict_emotion_single_file(model, audio_file_path):
    features_single_file = extract_features_single_file(audio_file_path)
    X_single = features_single_file.reshape((1, features_single_file.shape[0], 1))
    pred_single_file = model.predict(X_single)
    return pred_single_file


def predict_dominant_emotions(model, audio_file_path, encoder, chunk_size=2.5, overlap=0, top_n=2):
    data, sr = librosa.load(audio_file_path)
    chunk_samples = int(chunk_size * sr)
    overlap_samples = int(overlap * sr)
    emotion_predictions = []

    encoder_categories = encoder.categories_[0]
    emotions_list = encoder_categories.tolist()

    for i in range(0, len(data), chunk_samples - overlap_samples):
        chunk = data[i:i + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), 'constant')

        features = extract_features(chunk, sr)

        X = features.reshape((1, features.shape[0], 1))
        pred = model.predict(X)
        predicted_emotion = emotions_list[np.argmax(pred)]
        emotion_predictions.append(predicted_emotion)

    # Count emotion occurrences
    emotion_counts = Counter(emotion_predictions)

    # Get the top_n most dominant emotions
    top_emotions = emotion_counts.most_common(top_n)

    return top_emotions

Crema_df = prepare_dataset(Crema, crema_label_dict)
Ravdess_df = prepare_dataset(Ravdess, None)
Ravdess_df.replace({1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}, inplace=True)
Tess_df = prepare_dataset(Tess, tess_label_dict)
Savee_df = prepare_dataset(Savee, savee_label_dict)
comb_df = pd.concat([Ravdess_df, Tess_df, Crema_df, Savee_df], axis=0)
comb_df.to_csv('combined_df.csv', index=False)

# will drop rows with emotion "calm" because the amount of records less than 200
comb_df = comb_df[comb_df['Emotions'] != 'calm']
comb_df.reset_index(drop=True, inplace=True)
#print(comb_df['Emotions'].value_counts())

# X, Y = [], []
# for path, emotion in zip(comb_df['Path'], comb_df['Emotions']):
#     feature = get_features(path)
#     for ele in feature:
#         X.append(ele)
#         Y.append(emotion)
# Features = pd.DataFrame(X)
# Features['labels'] = Y
# Features.to_csv('features.csv', index=False)
Features = pd.read_csv('features.csv')

X = Features.iloc[:, :-1].values
Y = Features['labels'].values

encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42, shuffle=True)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# making our data compatible to model.
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
#print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# model = trained_model(x_train, y_train)
# print(model.summary())
#
# y_train_list = list(np.argmax(y_train, axis=1))
# class_weights = compute_class_weight('balanced', classes=np.unique(y_train_list), y=y_train_list)
# class_weights = dict(enumerate(class_weights))
# rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0001)
# history = model.fit(x_train, y_train, batch_size=128, epochs=4000, validation_data=(x_test, y_test), callbacks=[rlrp], class_weight=class_weights)
# model.save('my_model.h5')
# del model
#
# with open('training_history.json', 'w') as history_file:
#     json.dump(history.history, history_file)

# model = load_model('my_model.h5')
# with open('training_history.json', 'r') as history_file:
#      imported_history = json.load(history_file)

