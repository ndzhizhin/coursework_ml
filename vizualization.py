from imports import *



def generate_confusion_matrix(model, history, x_test, y_test, encoder):
    acc = "{:.2f}".format(model.evaluate(x_test, y_test)[1] * 100)

    epochs = range(len(history['loss']))

    fig, ax = plt.subplots(1, 2)
    train_acc = history['accuracy']
    train_loss = history['loss']
    test_acc = history['val_accuracy']
    test_loss = history['val_loss']

    fig.set_size_inches(20, 6)
    ax[0].plot(epochs, train_loss, label='Training Loss')
    ax[0].plot(epochs, test_loss, label='Testing Loss')
    ax[0].set_title('Training & Testing Loss')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")

    ax[1].plot(epochs, train_acc, label='Training Accuracy')
    ax[1].plot(epochs, test_acc, label='Testing Accuracy')
    ax[1].set_title('Training & Testing Accuracy')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")

    loss_img_data = io.BytesIO()
    fig.savefig(loss_img_data, format='png')
    plt.close(fig)
    loss_img_data.seek(0)
    loss_img_data_base64 = base64.b64encode(loss_img_data.getvalue()).decode('utf-8')

    pred_test = model.predict(x_test)
    y_pred = encoder.inverse_transform(pred_test)
    y_test = encoder.inverse_transform(y_test)

    cm_fig = plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    cm = pd.DataFrame(cm, index=[i for i in encoder.categories_], columns=[i for i in encoder.categories_])
    sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
    plt.title('Confusion Matrix', size=20)
    plt.xlabel('Predicted Labels', size=14)
    plt.ylabel('Actual Labels', size=14)

    cm_img_data = io.BytesIO()
    cm_fig.savefig(cm_img_data, format='png')
    plt.close(cm_fig)
    cm_img_data.seek(0)
    cm_img_data_base64 = base64.b64encode(cm_img_data.getvalue()).decode('utf-8')

    return loss_img_data_base64, cm_img_data_base64, acc


def get_classification_report_and_df(model, x_test, y_test, encoder):
    pred_test = model.predict(x_test)
    y_pred = encoder.inverse_transform(pred_test)
    y_test = encoder.inverse_transform(y_test)

    df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
    df['Predicted Labels'] = y_pred.flatten()
    df['Actual Labels'] = y_test.flatten()

    class_rep = classification_report(y_test, y_pred)

    return df, class_rep
