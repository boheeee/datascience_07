import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
           The categorical column 'pH' was temporarily encoded with one-hot encoding.
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

    # 범주형 데이터 pH 열에 대하여 더미 변수 생성(one-hot encoding) 함수 / pH 열은 지우고 해당 열을 one-hot encoding 방식으로 변환한다.
    # 이리하여 결측값이 많은 DissolvedOxygen (mg/L) (용존산소량) 열의 값을 회귀모델 예측치로 대체할 때 pH 열 또한 입력으로 넣을 수 있다.
    # Create a dummy variable for the pH column of categorical data (one-hot encoding) function

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

    # selec_dtypes 설명 필요
    # DissolvedOxygen (mg/L) 열을 제외하고 평균값으로 결측값 대체
    numeric_columns = df.select_dtypes(include='number').drop(
        'DissolvedOxygen (mg/L)', axis=1)
    df[numeric_columns.columns] = numeric_columns.fillna(
        numeric_columns.mean())

    # Missing values in the 'pH' column, which is a categorical column, are replaced with the mode value in that column
    mode_value = df['pH'].mode()[0]
    df['pH'].fillna(mode_value, inplace=True)

    # pH 열을 더미변수 (one-hot encoding) 방식으로 변환하여 용존 산소량 열을 제외한 모든 열을 사용하여 용존 산소량 결측치 값을 예측할 수 있다. 아래는 그 과정
    df_dummy = create_dummy_variables(df)

    # 전체 데이터에서 용존 산소량 값이 비어있지 않은 행들을 선택하여 학습 데이터 사용한다.
    df_train = df_dummy[df['DissolvedOxygen (mg/L)'].notnull()]
    # 마찬가지로 이번엔 비어있는 부분을 찾아서 해당 부분은 예측치 값을 채울 때 사용한다.
    df_missing = df_dummy[df_dummy['DissolvedOxygen (mg/L)'].isnull()]

    # 학습 데이터를 독립변수(입력)과 종속변수(타겟/예측 대상) 으로 구분
    X_train = df_train.drop(columns='DissolvedOxygen (mg/L)')
    y_train = df_train['DissolvedOxygen (mg/L)']

    # 선형 회귀 모델 학습
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # 용존 산소량 값이 비어있는 데이터에서 해당 열 삭제(타겟/예측 대상이기 때문)
    X_missing = df_missing.drop(columns='DissolvedOxygen (mg/L)')
    # 용존 산소량 결측치 예측/추정
    y_missing_pred = regressor.predict(X_missing)

    # 원본 데이터에서 결측치를 예측치로 대체
    df.loc[df['DissolvedOxygen (mg/L)'].isnull(),
           'DissolvedOxygen (mg/L)'] = y_missing_pred

    # 결측값을 처리한 깨끗한 데이터 반환
    return df


"""
    For a clean dataset, a function that returns a dictionary of 8 types of datasets 
    preprocessed with a combination of 4 types of scalers and 2 types of encoders.
    1) Scaler : {'min-max', 'standard', 'robust', 'mean(Custom)'}
    2) Encoder : {'one-hot', 'label'}
"""


def Create_Data_Preprocessing_Combinations(df_clean):

    # TermProject 자료(p27)에 있던 Mean normalization 스케일러를 직접 선언
    class Meannormalization(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            self.mean_ = np.mean(X, axis=0)
            self.min_ = np.min(X, axis=0)
            self.max_ = np.max(X, axis=0)
            return self

        def transform(self, X):
            return (X - self.mean_) / (self.max_ - self.min_)

        def fit_transform(self, X, y=None):
            return self.fit(X, y).transform(X)
    # 숫자형 열의 이름
    numeric_columns = df_clean.select_dtypes(
        include=['float64', 'int64']).columns
    # 범주형 열의 이름
    categorical_column = 'pH'

    # 전처리 조합에 사용될 스케일러들을 딕셔너리리로 생성
    scalers = {
        'min-max': MinMaxScaler(),
        'standard': StandardScaler(),
        'robust': RobustScaler(),
        'mean': Meannormalization()
    }
    # 전처리 조합에 사용될 인코더들을 딕셔너리로 생성
    encoders = {
        # sparse_output=False 이면 희소행렬로 출력하지 않게 한다. 기본값은 True 이며 이는 3개의 범주에 대하여 인코딩 시 하나의 열, (0,0,1) 이런 식의 출력물을 내놓는 것을 의미하며
        # False 일 시 각각을 새로운 열로 하여 0 | 0 | 1 로 생성한다. 여기서는 익숙한 방식으로 좀 더 다루기 쉽게하기 위해 값을 False 로 지정하여, pH 열에 대해 인코딩 하면 새로운 열 3개를 추가하도록 한다.
        'one-hot': OneHotEncoder(sparse_output=False),
        'label': LabelEncoder()
    }

    # 각 전처리 방법의 조합에 대한 데이터 프레임을 저장하기 위한 딕셔너리. key 값은 (스케일러 이름, 인코더 이름) 쌍의 튜플로 이루어지고 , value 는 DataFrame 이다.
    df_combinations = {}

    for scaler_name, scaler in scalers.items():
        # 범주형 데이터 'pH' 열을 제외한 나머지 열을 선택하여 복사
        scaled_df = df_clean.copy().drop(columns=[categorical_column])
        # 해당 열을 선택된 스케일러로 스케일링 한다
        scaled_df[numeric_columns] = scaler.fit_transform(
            scaled_df[numeric_columns])
        for encoder_name, encoder in encoders.items():
            # 범주형 데이터 'pH' 열을 제외한 나머지 열을 drop 한 뒤 복사, 즉 'pH' 열만 선택
            encoded_df = df_clean.copy().drop(columns=numeric_columns)
            # encoder 가 one-hot encoder 라면 pH 농도에 범주 3개에 대한 새로운 열 3개가 새로 생성되어야 한다.
            # get_feature_names_out 에 대해 조사해서 설명해주세요. 저도 잘 모르고 썼어요.
            if encoder_name == 'one-hot':
                encoded_df = pd.DataFrame(encoder.fit_transform(
                    encoded_df), columns=encoder.get_feature_names_out([categorical_column]))
            # label encoder 일 경우 기존 'pH' 열의 값을 인코딩 된 값으로 변경
            elif encoder_name == 'label':
                encoded_df[categorical_column] = encoder.fit_transform(
                    encoded_df[categorical_column])
            
            # 스케일링, 인코딩 한 데이터를 합쳐서 하나의 DataFrame 으로 만듬
            combined_df = pd.concat([scaled_df, encoded_df], axis=1)
            # 해당 값을 딕셔너리에 저장. ex) min-max scaler 와 label encoding 방식의 조합으로 생성된 데이터에 접근하기 위해서는 df_combinations[('min-max', 'label')]
            df_combinations[(scaler_name, encoder_name)] = combined_df

    # 조합으로 생성 된 8개의 value 를 갖는 딕셔너리 반환
    return df_combinations


df = pd.read_csv("data/waterquality.csv")
df_clean = Fill_Missing_Value(df)
df_combinations = Create_Data_Preprocessing_Combinations(df_clean)

# 모든 데이터 조합 출력
for key, value in df_combinations.items():
    print(f"{key}")
    print(value.head())


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
y_columns_one_hot = df_combinations[(
    'min-max', 'one-hot')].columns.tolist()[-3:]

best_combinations_score = {}

for (scaler_name, encoder_name), df in df_combinations.items():
    X = df.loc[:, X_columns]
    y = df.loc[:, y_columns_label] if encoder_name != 'one-hot' else df.loc[:,
                                                                            y_columns_one_hot]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7)

    knn_gscv.fit(X_train, y_train)
    tree_gscv.fit(X_train, y_train)
    if encoder_name == 'one-hot':
        logistic_gscv.fit(X_train, np.argmax(y_train.values, axis=1))
    else:
        logistic_gscv.fit(X_train, y_train)
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
