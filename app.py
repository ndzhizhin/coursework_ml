from imports import *
from main import predict_emotion_single_file, predict_dominant_emotions, x_test, y_test, encoder

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_audio_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('error.html', error='No file selected. Please choose a file and try again.')

        file = request.files['file']

        if file.filename == '':
            return render_template('error.html', error='No file selected. Please choose a file and try again.')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            try:
                audio_data, _ = librosa.load(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                duration = librosa.get_duration(y=audio_data)

                model = load_model('my_model.h5')
                with open('training_history.json', 'r') as history_file:
                    imported_history = json.load(history_file)

                encoder_categories = encoder.categories_[0]
                emotions_list = encoder_categories.tolist()

                if duration < 4:
                    pred = predict_emotion_single_file(model, os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    predicted_emotion = emotions_list[np.argmax(pred)]
                    emotions_str = predicted_emotion
                else:
                    top_emotions = predict_dominant_emotions(model, os.path.join(app.config['UPLOAD_FOLDER'], filename), encoder)
                    emotions_str = f"mostly {top_emotions[0][0]}, rarely {top_emotions[1][0]}"

                sentiment, intensity = analyze_sentiment(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                loss_img_data_base64, cm_img_data_base64, acc = generate_confusion_matrix(model, imported_history, x_test, y_test, encoder)

                df, class_rep = get_classification_report_and_df(model, x_test, y_test, encoder)

                results_df = pd.DataFrame({
                    'Emotion Prediction': [emotions_str],
                    'Sentiment': [sentiment],
                    'Intensity': [intensity],
                    'Accuracy on test data': [acc],
                })
                results_df.to_excel('results.xlsx', index=False)

                return render_template('results.html',
                                       predicted_emotion=emotions_str,
                                       sentiment=sentiment,
                                       intensity=intensity,
                                       acc=acc,
                                       loss_img_data_base64=loss_img_data_base64,
                                       cm_img_data_base64=cm_img_data_base64, df=df.head(10).to_html(), class_rep=class_rep)
            except Exception as e:
                return render_template('error.html', error=str(e))

    return render_template('upload.html')

@app.route('/download_results')
def get_download_link():
    return send_file('results.xlsx', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=False)