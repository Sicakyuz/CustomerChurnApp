import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc

# Add custom CSS for dark mode
def set_custom_style():
    st.markdown(
        """
        <style>
        body {
            background-color: #333333;
            color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: orange;
        }
        .st-eb {
            background-color: orange !important;
        }
        .st-dx {
            color: orange !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction Application",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get help": "https://www.streamlit.io/"}
)
# Cache data loading function
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    return cat_cols, num_cols, cat_but_car

def outliers_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr
    return low_limit, up_limit

def check_missing_values(dataframe):
    return dataframe.isnull().sum().any(), dataframe.isnull().sum()

def handle_outliers(dataframe, num_cols, q1=0.25, q3=0.75):
    for col in num_cols:
        low_limit, up_limit = outliers_thresholds(dataframe, col, q1, q3)
        dataframe.loc[(dataframe[col] < low_limit), col] = low_limit
        dataframe.loc[(dataframe[col] > up_limit), col] = up_limit

def preprocess_data(df):
    df = df.copy()
    #st.write("Initial data shape:", df.shape)

    # Handle missing values
    missing_values = df.isnull().sum()
    #st.write("Missing values per column:")
    #st.write(missing_values[missing_values > 0])

    threshold = 0.4 * len(df)
    df = df.dropna(thresh=threshold, axis=1)
    df = df.dropna()
    #st.write("Data shape after dropping NaNs:", df.shape)

    df["TotalCharges"] = df["TotalCharges"].replace(' ', np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
    df["TotalCharges"].fillna(df["TotalCharges"].mean(), inplace=True)

    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['ServiceCount'] = df[services].apply(lambda row: row[row == 'Yes'].count(), axis=1)
    df["gender"] = df["gender"].astype(str)

    df["FemaleSeniorCitizen"] = (df["gender"].str.lower() == "female").astype(int) * df["SeniorCitizen"]
    df["tenure"].replace(0, np.nan, inplace=True)
    df["tenure"].fillna(1, inplace=True)
    df['AverageMonthlyBill'] = df['TotalCharges'] / df['tenure']

    for service in services:
        df[f'{service}_CostPerService'] = df['TotalCharges'] * (df[service] == 'Yes')
    df['AverageServiceCost'] = df[[f'{service}_CostPerService' for service in services]].sum(axis=1)
    st.dataframe(df.head())
    if df.empty:
        st.error("The dataset is empty after preprocessing. Please check your data and try again.")
    return df

def encode_features(df):
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    label_encoder = LabelEncoder()
    for col in cat_cols:
        df[col] = label_encoder.fit_transform(df[col])
    return df
def num_summary(dataframe, numerical_col, plot=False):
    if dataframe[numerical_col].dtype in ['int64', 'float64']:  # Sadece sayısal sütunlar için işlem yap
        summary_df = dataframe[[numerical_col]].describe().transpose()

        st.subheader(f"Summary Statistics of {numerical_col}")
        styled_summary_df = summary_df.style.set_properties(**{'font-size': '16px', 'font-weight': 'bold'})
        st.dataframe(styled_summary_df)

        if plot:
            st.write(f"**Distribution of {numerical_col}**")
            color_key = f"Select Color for {numerical_col}"
            color = st.sidebar.color_picker(color_key, '#00f')
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(dataframe[numerical_col], bins=20, color=color, edgecolor='black', ax=ax)
            ax.set_xlabel(numerical_col, fontsize=14)
            ax.set_ylabel("Frequency", fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.warning("Please select a numerical variable.")

def cat_summary(dataframe, col_name, plot=False):
    if dataframe[col_name].dtype == 'object':  # Sadece kategorik sütunlar için işlem yap
        st.subheader(f"Summary Statistics of {col_name}")
        summary_df = pd.DataFrame({
            col_name: dataframe[col_name].value_counts(),
            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
        })
        styled_summary_df = summary_df.style.set_properties(**{'font-size': '16px'})
        st.table(styled_summary_df)

        if plot:
            st.sidebar.subheader("Graph Options")
            graph_types = st.sidebar.multiselect("Select Graph Types", ["Histogram", "Bar Chart", "Box Plot", "Violin Plot"])

            for graph_type in graph_types:
                st.write(f"**{graph_type}**")
                if graph_type == "Box Plot":
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(x=dataframe[col_name], ax=ax)
                    ax.set_xlabel(col_name, fontsize=14)
                    ax.set_ylabel("Value", fontsize=14)
                    ax.set_title(f"Box Plot of {col_name}", fontsize=16)
                    st.pyplot(fig)
                elif graph_type == "Violin Plot":
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.violinplot(x=dataframe[col_name], ax=ax)
                    ax.set_xlabel(col_name, fontsize=14)
                    ax.set_ylabel("Value", fontsize=14)
                    ax.set_title(f"Violin Plot of {col_name}", fontsize=16)
                    st.pyplot(fig)
                elif graph_type == "Histogram":
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(data=dataframe, x=col_name, bins=20, color='skyblue', edgecolor='black', ax=ax)
                    ax.set_xlabel(col_name, fontsize=14)
                    ax.set_ylabel("Frequency", fontsize=14)
                    ax.set_title(f"Histogram of {col_name}", fontsize=16)
                    st.pyplot(fig)
                elif graph_type == "Bar Chart":
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.countplot(data=dataframe, x=col_name, palette="Set3", ax=ax)
                    ax.set_xlabel(col_name, fontsize=14)
                    ax.set_ylabel("Count", fontsize=14)
                    ax.set_title(f"Bar Chart of {col_name}", fontsize=16)
                    plt.xticks(rotation=45)
                    for p in ax.patches:
                        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                    ha='center', va='bottom', fontsize=10, color='black')
                    st.pyplot(fig)
    else:
        st.warning("Please select categorical varible.")

def model_building(df):
    y = df["Churn"]
    X = df.drop(["Churn", "customerID"], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    log_model = LogisticRegression()
    log_model.fit(X_scaled, y)

    return log_model

def optimize_hyperparameters(df):
    y = df["Churn"]
    X = df.drop(["Churn", "customerID"], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'max_iter': [100, 200, 300, 400, 500]
    }
    log_model = LogisticRegression()
    grid_search = GridSearchCV(estimator=log_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_scaled, y)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    return best_model, best_params

def model_evaluation(model, df):
    y = df["Churn"]
    X = df.drop(["Churn", "customerID"], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc_value = auc(fpr, tpr)

    st.subheader("Classification Report")
    background_color = "#FFFFCC"  # Açık sarı
    text_color = "black"  # Siyah metin
        # Expander oluşturma
    with st.expander("View Classification Report Details"):
        st.markdown(f"""
            <div style='background-color: {background_color}; padding: 10px; border-radius: 10px; color: {text_color}'>
                <p><strong>Precision</strong>: The ratio of correctly predicted positive observations to the total predicted positives.</p>
                <p><strong>Recall</strong>: The ratio of correctly predicted positive observations to the all observations in actual class.</p>
                <p><strong>F1-score</strong>: The weighted average of Precision and Recall. It considers both false positives and false negatives.</p>                    <p><strong>Support</strong>: The number of occurrences of each class in y_true.</p>
                </div>
            """, unsafe_allow_html=True)
    st.text("Precision, Recall, F1-score")
    classification_df = pd.DataFrame(classification_rep).transpose()
    classification_df_styled = classification_df.style.format({
        'precision': "{:.2f}",
        'recall': "{:.2f}",
        'f1-score': "{:.2f}",
        'support': "{:.0f}"
    })
    st.table(classification_df_styled)


    st.subheader("ROC Curve")
    st.write("")
    with st.expander("**View ROC Curve Details**"):
        st.info("""
            The ROC curve is a graphical representation of the true positive rate (sensitivity) 
            against the false positive rate (1 - specificity) for different threshold values. 
            It helps evaluate the performance of a classification model across various 
            thresholds and is useful for comparing different models. 

            - AUC (Area Under the Curve) close to 1 indicates a very good model.
            - AUC around 0.5 suggests the model is not much better than random guessing.
            - AUC below 0.5 means the model is performing worse than random guessing.
        """)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc_value:.2f})')
    ax.plot([0, 1], [0, 1], linestyle='--', color='r')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc='lower right')
    st.pyplot(fig)


def plot_feature_importance(model, X_train):
    feature_names = X_train.columns
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': np.abs(model.coef_[0])})
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.title('Feature Importance', fontsize=16)
    st.pyplot(fig)


# Kullanımı



# Main Streamlit application
def main():
    # Set dark mode and primary color
    # Available themes: "light", "dark"
    # For primaryColor, use a valid CSS color name or code

    set_custom_style()

    # Sidebar'da navigasyon menüsü
    st.sidebar.markdown(
        "<style>div[role='radiogroup'] > label {margin-bottom: 10px;}</style>",
        unsafe_allow_html=True
    )

    navigation = st.sidebar.radio(
        "Go to",
        ("Upload Data", "Exploratory Data Analysis", "Model Building", "Hyperparameter Optimization",
         "Model Evaluation"),
        index=0,
        help="Select a section to navigate"
    )

    if navigation == "Upload Data":
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
        if uploaded_file:
            df = load_data(uploaded_file)
            st.success("Data Loaded Successfully!")
            st.write("First 5 rows of the data:")
            st.write(df.head())

    if navigation == "Exploratory Data Analysis":
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
        if uploaded_file:
            df = load_data(uploaded_file)
            df = preprocess_data(df)
            cat_cols, num_cols, cat_but_car = grab_col_names(df)
            # Sayısal veriler için özet ve dağılım grafiği seçenekleri
            numerical_col = st.sidebar.selectbox("Select Numerical Column", df.columns)
            show_num_summary = st.sidebar.checkbox("Show Numerical Summary", value=True)
            st.write("___")
            if show_num_summary:
                num_summary(df, numerical_col, plot=True)
            st.write("___")
            # Kategorik veriler için özet ve grafik seçenekleri
            categorical_col = st.sidebar.selectbox("Select Categorical Column", df.columns)
            show_cat_summary = st.sidebar.checkbox("Show Categorical Summary", value=True)
            if show_cat_summary:
                cat_summary(df, categorical_col, plot=True)
            st.write("___")

    if navigation == "Model Building":
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
        if uploaded_file:
            df = load_data(uploaded_file)
            df = preprocess_data(df)
            df = encode_features(df)
            model = model_building(df)
            st.success("Logistic Regression Model Built Successfully!")
            st.write("Model Coefficients:")

            st.write()
            coef_df = pd.DataFrame(
                {"Feature": df.drop(["Churn", "customerID"], axis=1).columns, "Coefficient": model.coef_[0]})
            coef_df_transposed = coef_df.T
            st.write(coef_df_transposed)

    if navigation == "Hyperparameter Optimization":
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
        if uploaded_file:
            df = load_data(uploaded_file)
            df = preprocess_data(df)
            df = encode_features(df)
            best_model, best_params = optimize_hyperparameters(df)
            st.success("Hyperparameter Optimization Completed Successfully!")
            st.subheader("Best Parameters:")
            st.table(pd.DataFrame(best_params.items(), columns=['Parameter', 'Value']))
            st.subheader("Model Coefficients and Feature Importance")
            st.write("Best Model Coefficients:")
            coef_df = pd.DataFrame(
                {"Feature": df.drop(["Churn", "customerID"], axis=1).columns, "Coefficient": best_model.coef_[0]})
            coef_df_transposed = coef_df.T
            st.write(coef_df_transposed)
            plot_feature_importance(best_model, df.drop(["Churn", "customerID"], axis=1))

    if navigation == "Model Evaluation":
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
        if uploaded_file:
            df = load_data(uploaded_file)
            df = preprocess_data(df)
            df = encode_features(df)
            model = model_building(df)
            model_evaluation(model, df)
            st.sidebar.title("")
            st.sidebar.title("")
            st.sidebar.title("")
            st.sidebar.title("")
            st.sidebar.header("Make Prediction for New Customer")
            st.sidebar.write(f"Total Number of Observations: {len(df)}")
            selected_index = st.sidebar.number_input("Select an index for prediction", min_value=0,
                                                     max_value=len(df) - 1,
                                                     step=1)
            new_customer_data = df.iloc[selected_index].drop(["Churn", "customerID"])
            st.sidebar.write("Selected Customer Data for Prediction:")
            st.sidebar.write(new_customer_data)
            if st.sidebar.button("Predict"):
                new_customer_data = new_customer_data.values.reshape(1, -1)
                prediction = model.predict(new_customer_data)
                st.sidebar.write("Prediction Result:")
                if prediction[0] == 1:
                    st.sidebar.write("**Churn: Yes**")
                else:
                    st.sidebar.write("**Churn: No**")


if __name__ == "__main__":
    main()
