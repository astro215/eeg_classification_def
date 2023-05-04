import streamlit as st
import mne
import os
import numpy as np
import pandas as pd
import joblib
from scipy import stats
import sklearn
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from PIL import Image

st.set_page_config(layout="wide", page_title="EEG Classification", page_icon=":brain:")

# load assets
sml_poster = Image.open("poster/SML_EEG.png")
sml_ppt = Image.open("poster/SML_ProjectReview.png")
dataset_img = Image.open("poster/data.png")
about_img = Image.open("poster/about.png")


# define a function to read the data from the .set file
def read_data(file_path):
    data = mne.io.read_raw_eeglab(file_path, preload=True)
    data.apply_function(lambda x: np.convolve(x, np.ones(50) / 50, mode='same'))
    data.set_eeg_reference()
    data.filter(l_freq=1, h_freq=50)
    epochs = mne.make_fixed_length_epochs(data, duration=1, preload=True)
    array = epochs.get_data()
    return array


def pca_to_epochs(file_path, n_components=15, epoch_length=1, sfreq=250):
    # eeg_data is a numpy array of shape (n_channels, n_samples)
    eeg_data = mne.io.read_raw_eeglab(file_path, preload=True)
    eeg_data.apply_function(lambda x: np.convolve(x, np.ones(50) / 50, mode='same'))
    eeg_data.set_eeg_reference()
    eeg_data.filter(None, 50., fir_design='firwin')
    # Apply PCA to the EEG data
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(eeg_data.get_data().T)  # Transpose data to shape (n_samples, n_channels) for PCA

    # Create an MNE RawArray object from the PCA data
    channel_names = ['PC{}'.format(i + 1) for i in range(n_components)]
    info = mne.create_info(channel_names, sfreq, ch_types='eeg')
    raw_pca = mne.io.RawArray(pca_data.T, info)

    # Convert the RawArray to epochs
    tmax = epoch_length - 1 / sfreq  # Adjust tmax to account for last sample in epoch
    events = mne.make_fixed_length_events(raw_pca, duration=epoch_length)
    epochs = mne.Epochs(raw_pca, events=events, tmin=0, tmax=tmax, baseline=None, preload=True)

    return epochs.get_data()


def ica_to_epochs(file_path, n_components=12, epoch_length=1, sfreq=250):
    # eeg_data is a numpy array of shape (n_channels, n_samples)
    eeg_data = mne.io.read_raw_eeglab(file_path , preload=True)
    eeg_data.apply_function(lambda x: np.convolve(x, np.ones(50)/50, mode='same'))
    eeg_data.set_eeg_reference()
    eeg_data.filter(None, 50., fir_design='firwin')
    # Apply ICA to the EEG data
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=97)
    ica.fit(eeg_data)
    # Apply the ICA to the raw data
    raw_ica = ica.apply(eeg_data.copy(), exclude=ica.exclude)

    # Convert the RawArray to epochs
    tmax = epoch_length - 1/sfreq  # Adjust tmax to account for last sample in epoch
    events = mne.make_fixed_length_events(raw_ica, duration=epoch_length)
    epochs = mne.Epochs(raw_ica, events=events, tmin=0, tmax=tmax, baseline=None, preload=True)

    return epochs.get_data()


def mean(x):
    return np.mean(x, axis=-1)


def std(x):
    return np.std(x, axis=-1)


def ptp(x):
    return np.ptp(x, axis=-1)


def skew(x):
    return stats.skew(x, axis=-1)


def var(x):
    return np.var(x, axis=-1)


def min(x):
    return np.min(x, axis=-1)


def max(x):
    return np.max(x, axis=-1)


def argmin(x):
    return np.argmin(x, axis=-1)


def argmax(x):
    return np.argmax(x, axis=-1)


def rms(x):
    return np.sqrt(np.mean(x ** 2, axis=-1))


def abs_diff_signal(x):
    return np.sum(np.abs(np.diff(x, axis=-1)), axis=-1)


def kurtosis(x):
    return stats.kurtosis(x, axis=-1)


def concatenate_fucntions(x):
    return np.concatenate((mean(x), std(x), ptp(x), skew(x), var(x), min(x), max(x), rms(x), argmin(x), argmax(x),
                           abs_diff_signal(x), kurtosis(x)), axis=-1)


def model():
    uploaded_file = st.file_uploader("Upload a .set file", type=".set")

    if uploaded_file is not None:
        # save the uploaded file to disk
        file_path = os.path.join("./", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # read the data from the uploaded file

        st.header("Customizing the model.")
        model_name_display = ["Logistic Regression", "KNN", "Random Forest", "Decision Tree", "XGBoost"]
        model_ = st.selectbox("Select the model",model_name_display)
        # get model index from model name
        model_index = model_name_display.index(model_)
        # create a radio button to select between ICA, PCA, and None
        model_names = ['logReg.pkl', 'knn.pkl', 'randomForest.pkl', 'decisionTree.pkl', 'xgboost.pkl']

        # load the selected model
        model_name = model_names[model_index]

        # create a radio button to select between ICA, PCA, and None
        dimension_reduction = st.radio("Select a dimension reduction method", ("ICA", "PCA", "None"))

        # load the selected model and the corresponding dimension reduction method

        if st.button("Predict"):
            if dimension_reduction == "ICA":
                data = ica_to_epochs(file_path)
                features_array = np.array(concatenate_fucntions(data))
                model_path = "models/" + model_name[:-4] + "_ica.pkl"
            elif dimension_reduction == "PCA":
                data = pca_to_epochs(file_path)
                features_array = np.array(concatenate_fucntions(data))
                model_path = "models/" + model_name[:-4] + "_pca.pkl"
            else:
                data = read_data(file_path)
                features_array = np.array(concatenate_fucntions(data))
                model_path = "models/" + model_name

            with open(model_path, 'rb') as f:
                model = joblib.load(f)

            dict_pred = {0: "Eyes Open", 1: "Eyes Closed", 2: "Memory Task", 3: "Music Task", 4: "Math Task"}
            result = model.predict(features_array)

            # data frame with features and results labels
            df = pd.DataFrame(features_array)
            df["predicted"] = result
            # map labels to emotions
            df["predicted"] = df["predicted"].map(dict_pred)

            # create data frames using value_counts
            df1 = df["predicted"].value_counts().rename_axis('predicted').reset_index(name='count')

            st.header("Predicted Labels")
            st.subheader(model_ + " with " + dimension_reduction)
            st.dataframe(df1)

            mode = df1["count"].idxmax()
            st.write("Predicted class :", df1["predicted"][mode])


def home_page():
    st.title("EEG Signal Classification")
    with st.container():
        st.subheader("Introduction")
        st.write(
            "EEG signals (electroencephalogram signals) are electrical brain signals that are recorded by placing electrodes on the scalp. These signals are generated by the electrical activity of neurons in the brain and can be used to study brain function, cognitive processes, and various neurological disorders. EEG signals are measured as voltage fluctuations over time, and can provide information about the timing, frequency, and amplitude of brain activity in different regions of the brain. EEG signals can be recorded during different states such as resting, cognitive tasks, sleep, and other activities, providing insights into the underlying neural processes.")
        st.write("---")
        st.subheader("Key Features")
        st.write(
            """
            - This is a web application that classifies EEG signals into 5 classes: Eyes Open, Eyes Closed, Memory Task Music Task, and Math Task.
            - The application uses a machine learning model trained on the EEG data from 100 subjects. The model is trained on  features extracted from the EEG signals using ICA and PCA.
            - We have trained 5 different machine learning models: Logistic Regression, KNN, Random Forest, Decision Tree, and XGBoost.
            - The application allows the user to upload a .set file containing EEG data and predict the class of the EEG signal.""")
        st.write("---")
        st.subheader("Problem Statement")
        st.write("To predict a specific outcome variable, such as sleep quality, emotion, "
                 "mental health, or mind-wandering tendencies based on the EEG and other measures to build a "
                 "supervised machine learning model that can accurately classify EEG recordings from 60 participants "
                 "into one of the five classes: resting with eyes closed, resting with eyes open, cognitive task of "
                 "subtraction, cognitive task of listening to music, and cognitive task of memory.")
        st.write("---")
        st.subheader("Future Scope")
        st.write(
            "Using EEG signals to map dopamine and addiction by allowing subjects to perform certain tasks when the mind is tired and releasing dopamine by allowing them to do dopamine releasing activities, and then performing the same tasks which they were finding difficult to before consuming drinkables/eatables which induce dopamine or performing tasks which release dopamine. By extending this research we can identify how distraction when consumed more can hamper cognitive behaviour and neural ability, and therefore we can take this research and generate certain novel insights which would aware and benefit the society.")
        with st.container():
            st.write("---")
            st.write("##")
            image_column, text_column = st.columns((1, 2))
            with image_column:
                st.image(sml_poster, use_column_width=True)
            with text_column:
                st.subheader("EEG Classification - Machine Learning")
                st.write(
                    """
                    This poster explains and summarizes the project. 
                    """
                )
                st.markdown(
                    "[Open](https://www.canva.com/design/DAFhB5pisT0/Xx_WGXhZhB6OWcUWLVNpHw/edit?utm_content=DAFhB5pisT0&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)")

            with st.container():
                st.write("---")
                st.write("##")
                image_column, text_column = st.columns((1, 2))
                with image_column:
                    st.image(sml_ppt, use_column_width=True)
                with text_column:
                    st.subheader("EEG Classification - PPT")
                    st.write(
                        """
                        This PPT explains the overall project in brief. 
                        """
                    )
                    link_str = "https://www.canva.com/design/DAFdMZLx4zo/LvIN87ZujuIKXT8FkFjIrA/edit?utm_content=DAFdMZLx4zo&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton"
                    # link_str2 = "https://drive.google.com/drive/folders/1cFb_WIXBSvzkGFMEtjxAtnz502aEXSM4?usp=sharing"
                    st.markdown(f"[View]({link_str})")

            with st.container():
                st.write("---")
                st.write("##")
                image_column, text_column = st.columns((1, 2))
                with image_column:
                    st.image(dataset_img)
                with text_column:
                    st.subheader("Dataset - OpenNeuro")
                    st.write(
                        """
                        The dataset includes electroencephalogram (EEG) data from 60 participants with all three recording sessions, including the present (session 1), 90 min later (session 2), and one month later (session 3). The average age of all the participants is 20.01 years old (range 18–28) and the median is 20 years old. There are 32 females and 28 males. Part of the dataset was utilized to investigate the reproducibility of power spectrum, functional connectivity and network construction in eyes-open and eyes-closed resting-state EEG, and was published in Journal of Neuroscience Methods3. 
                        """
                    )
                    link_str = "https://openneuro.org/datasets/ds004148/versions/1.0.1"
                    st.markdown(f"[View]({link_str})")


# create a Streamlit app
def about_us():
    st.image(about_img)


def app():
    tab1, tab2, tab3 = st.tabs(["Our Project", "Model", "About us"])
    with tab1:
        home_page()
    with tab2:
        model()
    with tab3:
        about_us()


# run the app
if __name__ == '__main__':
    app()
