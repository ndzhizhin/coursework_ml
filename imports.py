import pandas as pd
import numpy as np
import os
import random
import math
import sys
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, Flatten, Dropout, BatchNormalization, Activation, GlobalMaxPooling1D
from keras.layers import Input, LSTM, Bidirectional, Attention
from keras.callbacks import ReduceLROnPlateau
import flask
from flask import Flask, render_template, request, redirect, url_for, send_file
import io
import base64
import xlsxwriter
from werkzeug.utils import secure_filename
from data_processing import prepare_dataset, data_augmentation, extract_features, get_features, extract_features_single_file, analyze_sentiment
from model_training import trained_model
from vizualization import generate_confusion_matrix, get_classification_report_and_df
import json
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

