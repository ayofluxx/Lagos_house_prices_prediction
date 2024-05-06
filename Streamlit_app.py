import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor


st.write("""

# Lagos Rent price Prediction

This app predict rent price
""")

st.sidebar.header('User Input Features')


def user_features():
    city = st.sidebar.text_input('Enter City name ')
    neighborhood = st.sidebar.text_input('Enter Neighborhood ')
    newly_built = st.sidebar.selectbox('Is it Newly_built ?',(0, 1))
    furnished = st.sidebar.selectbox('Is it Furnished ?',(0, 1))
    bedrooms = st.sidebar.text_input('How many Bedrooms ? ')
    bathrooms = st.sidebar.text_input('How many Bathrooms ? ')
    toilets = st.sidebar.text_input('How many Toilets ? ')

    data = {
        'city':  city,
        'neighborhood': neighborhood,
        'newly_built': newly_built,
        'furnished': furnished,
        'bedrooms':  bedrooms,
        'bathrooms':  bathrooms,
        'toilets': toilets
            }
    features = pd.DataFrame(data, index=[0])
    return features

CHUNKSIZE = 10000  # Number of rows to read at a time
for chunk in pd.read_csv('cleandf.csv', chunksize=CHUNKSIZE):
    lagos_df = chunk

lagos_df = pd.read_csv('/Users/4yoboy/PycharmProjects/Streamlit/cleandf.csv')
X = lagos_df.drop(['price'], axis = 1)
y = lagos_df['price']

# Convert 'y' to a NumPy array and reshape
y_array = np.array(y).reshape(-1, 1)

# Apply Box-Cox transformation to the target variable
pt = PowerTransformer(method='box-cox')
y_transformed = pt.fit_transform(y_array).flatten()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.1, random_state = 123)

sv = SVR(kernel = 'rbf')
sv_ = TransformedTargetRegressor(regressor=sv, transformer=PowerTransformer(), check_inverse=False)
sv_.fit(X_train, y_train)


def predict_price(features_df):
    try:
        city_index = np.where(X_test.columns == features_df['city'].values[0])[0][0]
        nei_index = np.where(X_test.columns == features_df['neighborhood'].values[0])[0][0]
    except IndexError:
        print(f"City '{features_df['city'].values[0]}' or Neighborhood '{features_df['neighborhood'].values[0]}' not found in columns.")
        return None

    x = np.zeros(len(X_test.columns))
    x[0] = features_df['newly_built'].values[0]
    x[1] = features_df['furnished'].values[0]
    x[2] = features_df['bedrooms'].values[0]
    x[3] = features_df['bathrooms'].values[0]
    x[4] = features_df['toilets'].values[0]
    if city_index >= 0:
        x[city_index] = 1
    if nei_index >= 0:
        x[nei_index] = 1

    prediction_transformed = sv_.predict([x])[0]
    prediction_actual = pt.inverse_transform([[prediction_transformed]])[0, 0]

    return prediction_actual



input_df = user_features()
lagos_df  = pd.read_csv('cleandf.csv')
lagos = lagos_df.drop(['price'], axis = 1)
df = pd.concat([input_df,lagos], axis = 0)







# Apply model to make predictions
prediction = predict_price(df)

st.subheader('Predicted Rent Price')
st.write(prediction)



