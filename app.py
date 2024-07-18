import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import pickle


def preprocess_data(df):
    df.dropna(inplace=True)
    df2 = df[df["impluse"] < 1000]
    if df2["class"].dtypes == "object":
        label = {"positive": 1, "negative": 0}
        df2["class"] = df2["class"].map(label)
    return df2


def load_model(model):
    with open(f"./model/{model}", "rb") as file:
        model = pickle.load(file)

    return model


def plot_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    label = {"positive": 1, "negative": 0}

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)

    st.subheader("Total label count")
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax = y_test.value_counts().plot(kind="bar")
    st.pyplot(fig)

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        fmt="d",
        annot=True,
        cmap="Blues",
        xticklabels=label.keys(),
        yticklabels=label.keys(),
        cbar=False,
        square=True,
        ax=ax,
    )
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Area Under the Curve (AUC)")
    st.write(f"AUC: {roc_auc}")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", label=f"ROC curve (area = {roc_auc:.2f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    st.pyplot(fig)

    st.subheader("Precision-Recall Curve")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="blue")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    st.pyplot(fig)


def home():
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    st.write("Upload file with this format:")
    st.image("images/file_format.jpg")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            df = preprocess_data(df)

            model = load_model("model.pkl")

            X = df.drop(columns="class")
            y = df["class"]

            plot_model(model, X, y)

        except:
            st.error("Please upload a valid dataset.")


def predict(model, input_data):
    age = input_data["age"]
    gender = input_data["gender"]
    impulse = input_data["impulse"]
    pressure_high = input_data["pressure_high"]
    pressure_low = input_data["pressure_low"]
    glucose = input_data["glucose"]
    kcm = input_data["kcm"]
    troponin = input_data["troponin"]

    features = [
        age,
        gender,
        impulse,
        pressure_high,
        pressure_low,
        glucose,
        kcm,
        troponin,
    ]

    prediction = model.predict([features])[0]

    return prediction


def inference():
    st.header("Heart disease Prediciton")

    st.sidebar.header("User Inputs")
    age = st.sidebar.number_input("Age ", min_value=0, max_value=120, value=30)
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    gender = 1 if gender == "Male" else 0
    impulse = st.sidebar.number_input("Impulse", min_value=20, max_value=200, value=78)
    pressure_high = st.sidebar.number_input(
        "High Blood Pressure", min_value=40, max_value=225, value=120
    )
    pressure_low = st.sidebar.number_input(
        "Low Blood Pressure", min_value=35, max_value=155, value=80
    )
    glucose = st.sidebar.number_input(
        "Glucose", min_value=30.0, max_value=550.0, value=146.0, step=1.0
    )
    kcm = st.sidebar.number_input(
        "KCM", min_value=0.0, max_value=300.0, value=2.0, step=1.0
    )
    troponin = st.sidebar.number_input(
        "Troponin", min_value=0.0, max_value=10.0, value=0.01, step=0.1
    )

    user_input = {
        "age": age,
        "gender": gender,
        "impulse": impulse,
        "pressure_high": pressure_high,
        "pressure_low": pressure_low,
        "glucose": glucose,
        "kcm": kcm,
        "troponin": troponin,
    }

    model = load_model("model.pkl")

    if st.sidebar.button("Predict"):
        prediction = predict(model, user_input)
        if prediction == 1:
            st.success("Condition: Positive")
        else:
            st.error("Condition: Negative")


def main():
    st.title("Streamlit demo app")

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Home", "Inference"])

    if selection == "Home":
        home()
    elif selection == "Inference":
        inference()


if __name__ == "__main__":
    main()
