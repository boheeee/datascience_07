import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier


"""
    Functions to fill in missing values in data. 
    1. delete unnecessary columns: 'Date'
    2. create a categorical data attribute: 'pH' (float -> categorical)
    3. fill in the missing values 
        1) Replace with the mean value for real columns
        2) Replace with the mode value for categorical columns
        3) Predict with a regression model for the column 'DissolvedOxygen (mg/L)', which has many missing values:
           The categorical column 'pH' was temporarily encoded with one-hot encoding for regression
"""


def Fill_Missing_Value(df):
    # Delete the unnecessary 'Date' column
    df.drop(columns=['Date'], inplace=True)

    """
        Number of samples before removing missing values
        print(df.shape[0]) = 2371

        number of samples after removing rows containing missing values
        df_clean = df.dropna()
        print(df_clean.shape[0]) = 1320

        if we remove all rows with missing values, we would lose about 45% of the data, potentially losing representativeness
        so rather than use a regression/classification model to fill in the missing values, we decided to use the mean/mode values
    """

    # Functions to convert real pH values to categorical data

    def categorize_pH(ph_value):
        if pd.isna(ph_value):
            return np.nan  # Leave missing values as iss
        elif ph_value == 7.0:
            return 'Neutral'
        elif ph_value < 7.0:
            return 'Acidic'
        else:
            return 'Alkaline'

    """
        Create a dummy variable for the categorical data pH column (one-hot encoding) function
        Delete the pH column and convert the column to one-hot encoding.
        This way, when replacing the values in the DissolvedOxygen (mg/L) column with regression model predictions, 
        which have many missing values, the pH column can also be included as input.
    """
    def create_dummy_variables(df):
        dummies = pd.get_dummies(df['pH'], prefix='pH')
        df = df.drop('pH', axis=1)
        df = pd.concat([df, dummies], axis=1)
        return df

    df['pH'] = df['pH'].apply(categorize_pH)
    df['pH'] = df['pH'].astype('category')

    # Output the number of missing values
    missing_values_count = df.isnull().sum()
    print("The number of missing values in each column:")
    print(missing_values_count)
    """
        For the 'DissolvedOxygen (mg/L)' column, the number of missing values is 851, which is about 35% of the total data.
        Because of this, we decided to replace the missing values in the other columns with the mean/mode 
        and then build a regression model using these data to get a prediction. 
    """

    # Output the number of counts for each category in the pH column
    ph_value_counts = df['pH'].value_counts(dropna=False)
    print("\nThe number of each category in the 'pH' column:")
    print(ph_value_counts)

    # Replace missing values with mean values, except for the 'DissolvedOxygen (mg/L)' column
    numeric_columns = df.select_dtypes(include='number').drop(
        'DissolvedOxygen (mg/L)', axis=1)
    df[numeric_columns.columns] = numeric_columns.fillna(
        numeric_columns.mean())

    # Missing values in the 'pH' column, which is a categorical column, are replaced with the mode value in that column
    mode_value = df['pH'].mode()[0]
    df['pH'].fillna(mode_value, inplace=True)

    # Convert a column of categorical data to a dummy variable (= One-hot encoding)
    df_dummy = create_dummy_variables(df)

    # Select rows with non-empty 'DissolvedOxygen (mg/L)' values from the entire data to use as training data.
    df_train = df_dummy[df['DissolvedOxygen (mg/L)'].notnull()]
    # Similarly, this time we look for empty parts and use them to fill in the predicted value.
    df_missing = df_dummy[df_dummy['DissolvedOxygen (mg/L)'].isnull()]

    # Divide training data into independent variables (inputs) and dependent variables (targets/predictions)
    X_train = df_train.drop(columns='DissolvedOxygen (mg/L)')
    y_train = df_train['DissolvedOxygen (mg/L)']

    # Train a linear regression model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Delete the column from the data where the 'DissolvedOxygen (mg/L)' value is empty (because it is targeted/predicted)
    X_missing = df_missing.drop(columns='DissolvedOxygen (mg/L)')
    # Predicting 'DissolvedOxygen (mg/L)' missing values
    y_missing_pred = regressor.predict(X_missing)

    # Replacing missing values in original data with predictions
    df.loc[df['DissolvedOxygen (mg/L)'].isnull(),
           'DissolvedOxygen (mg/L)'] = y_missing_pred

    # Return clean data with missing values handled
    return df


"""
    For a clean dataset, a function that returns a dictionary of 8 types of datasets 
    preprocessed with a combination of 4 types of scalers and 2 types of encoders.
    1) Scaler : {'min-max', 'standard', 'robust', 'mean(Custom)'}
    2) Encoder : {'one-hot', 'label'}
"""


def Create_Data_Preprocessing_Combinations(df_clean):

    # Defining the Mean normalisation class
    class Meannormalization():
        def fit(self, X, y=None):
            self.mean_ = np.mean(X, axis=0)
            self.min_ = np.min(X, axis=0)
            self.max_ = np.max(X, axis=0)
            return self

        def transform(self, X):
            return (X - self.mean_) / (self.max_ - self.min_)

        def fit_transform(self, X, y=None):
            return self.fit(X, y).transform(X)

    # Names for numeric columns
    numeric_columns = df_clean.select_dtypes(
        include=['float64', 'int64']).columns
    # Names for categorical columns
    categorical_column = 'pH'

    # Create a dictionary of scalers to be used in preprocessing combinations
    scalers = {
        'min-max': MinMaxScaler(),
        'standard': StandardScaler(),
        'robust': RobustScaler(),
        'mean': Meannormalization()
    }
    # Create a dictionary of encoders to be used in preprocessing combinations
    encoders = {
        """
            If sparse_output=False, do not output to sparse matrix. 
            The default is True, which means that the encoding will output one column for each of the three categories, (0,0,1), and so on.
            If False, it will generate 0 | 0 | 1 with each as a new column. 
            Here, to make it more tractable in a familiar way, we'll specify the value as False, 
            so that encoding for the pH column will add three new columns.
        """
        'one-hot': OneHotEncoder(sparse_output=False),
        'label': LabelEncoder()
    }
    """
        A dictionary for storing data frames for each combination of preprocessing methods. 
        The key values are tuples of (scaler name, encoder name) pairs, and the value is a DataFrame.
    """
    df_combinations = {}

    for scaler_name, scaler in scalers.items():
        # Select and copy the remaining columns except the categorical data 'pH' column
        scaled_df = df_clean.copy().drop(columns=[categorical_column])
        # Scales the column by the selected scaler
        scaled_df[numeric_columns] = scaler.fit_transform(
            scaled_df[numeric_columns])
        for encoder_name, encoder in encoders.items():
            # Categorical data drop and copy the remaining columns except for the 'pH' column, i.e. select only the 'pH' column
            encoded_df = df_clean.copy().drop(columns=numeric_columns)
            # If the encoder is a one-hot encoder, then the pH concentration should have three new columns for the three categories.
            if encoder_name == 'one-hot':
                encoded_df = pd.DataFrame(encoder.fit_transform(
                    encoded_df), columns=encoder.get_feature_names_out([categorical_column]))
            
            # Change the value of the existing 'pH' column to the encoded value if label encoder
            elif encoder_name == 'label':
                encoded_df[categorical_column] = encoder.fit_transform(
                    encoded_df[categorical_column])

            # Scaling and combining encoded data into a single DataFrame
            combined_df = pd.concat([scaled_df, encoded_df], axis=1)
            df_combinations[(scaler_name, encoder_name)] = combined_df

    # Return a dictionary with 8 values generated by the combination
    return df_combinations


def Visualise_data_distribution(df):
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(df.columns, start=1):
        if col == 'pH':
            pH_counts = df[col].value_counts()
            plt.subplot(3, 3, i)
            plt.pie(pH_counts, labels=pH_counts.index, autopct='%1.1f%%')
            plt.title(f"Distribution of {col}")
        else:
            plt.subplot(3, 3, i)
            sns.kdeplot(df[col], fill=True)
            plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.show()


df = pd.read_csv("data/waterquality.csv")
df_clean = Fill_Missing_Value(df)
Visualise_data_distribution(df_clean)
df_combinations = Create_Data_Preprocessing_Combinations(df_clean)

# Output all data combinations
for key, value in df_combinations.items():
    print(f"{key}")
    print(value.head(), "\n")


# 사용할 모델과 hyperparameter tuning 에 사용될 parameter 를 딕셔너리(grid) 형식으로 지정
# 아래의 parameter 들에 대해 조사하여 문서 작성 시 추가해 주세요. 예를 들면 C 는 로지스틱스 회귀 모델에서 규제 정도를 나타내는 값입니다.
# 특히 여기서는 사용되지 않았지만 LogisticRegression() 의 parameter 로 penalty 와 solver 가 있는데 특정 solver 는 특정 penalty 에서만 유효하게 동작한다고 하는데 왜 그런지 간단하게 조사하면 좋을 거 같아요
knn_param_grid = {
    'n_neighbors': np.arange(1, 29),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}
tree_param_grid = {
    'criterion': ["gini", "entropy", "log_loss"],
    'splitter': ["best", "random"],
    'max_depth': np.arange(5, 20)
}
logistic_param_grid = {
    'C': np.logspace(-4, 4, 20),
    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
}

# 각각의 모델과 모델별 parameter 조합, cv(cross validation) 값은 10으로 고정, 으로 하여 GridSearchCV 모델 생성
knn_gscv = GridSearchCV(KNeighborsClassifier(),
                        param_grid=knn_param_grid, cv=10)
tree_gscv = GridSearchCV(DecisionTreeClassifier(),
                         param_grid=tree_param_grid, cv=10)
logistic_gscv = GridSearchCV(
    LogisticRegression(), param_grid=logistic_param_grid, cv=10)

X_columns = df_clean.select_dtypes(
    include=['float64', 'int64']).columns
y_columns_label = 'pH'
y_columns_one_hot = ['pH_Acidic',  'pH_Alkaline',  'pH_Neutral']

best_combinations_score = {}

for (scaler_name, encoder_name), df in df_combinations.items():
    X = df.loc[:, X_columns]
    y = df.loc[:, y_columns_label] if encoder_name != 'one-hot' else df.loc[:,
                                                                            y_columns_one_hot]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7)

    knn_gscv.fit(X_train, y_train)
    tree_gscv.fit(X_train, y_train)
    logistic_gscv.fit(X_train, y_train if encoder_name !=
                      'one-hot' else np.argmax(y_train.values, axis=1))

    print(f"\nFor a dataset with scaler({
          scaler_name}) and encoder({encoder_name}) :")

    print(
        f"knn - best parameter : {knn_gscv.best_params_} / best score : {knn_gscv.best_score_} ")
    # zero_division 에 대해서도 설명 부탁드려요. 대충 알기론 TP/(TP+FP) 에서 한 번도 예측되지 않은 클래스가 있다면 분모가 0이 되기 때문에 이를 방지하기 위한 거라고만 알고 있는데 문서 작성 시 좀 더 자세히
    print(classification_report(
        y_test, knn_gscv.best_estimator_.predict(X_test), zero_division=1))

    print(
        f"tree - best parameter : {tree_gscv.best_params_} / best score : {tree_gscv.best_score_} ")
    print(classification_report(
        y_test, tree_gscv.best_estimator_.predict(X_test), zero_division=1))

    print(f"logistic - best parameter : "
          f"{logistic_gscv.best_params_} / best score : {logistic_gscv.best_score_} \n")
    print(classification_report(y_test if encoder_name != 'one-hot' else np.argmax(y_test.values,
          axis=1), logistic_gscv.best_estimator_.predict(X_test), zero_division=1))

    best_combinations_score[(
        f'{scaler_name}/{encoder_name}', 'knn')] = knn_gscv.best_score_
    best_combinations_score[(
        f'{scaler_name}/{encoder_name}', 'tree')] = tree_gscv.best_score_
    best_combinations_score[(
        f'{scaler_name}/{encoder_name}', 'logistic')] = logistic_gscv.best_score_

sorted_dic = sorted(best_combinations_score.items(),
                    key=lambda x: x[1], reverse=True)
for key, value in sorted_dic:
    print(f"{key} : {value}")
