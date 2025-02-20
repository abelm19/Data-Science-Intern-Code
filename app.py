#Import needed libraries
import numpy as np
import pandas as pd
import pickle
import streamlit as st
from PIL import Image
import plotly.graph_objects as go  # For interactive plots

# Reading the csv file for making historical vs prediction plots
df = pd.read_csv('Global_Temp - Global_Temp.csv.csv')
df2 = df.drop('Year', axis = 1)
df2['Year'] = df['Year']
input_data = df2[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
       'Oct', 'Nov', 'Dec', 'D-N', 'DJF', 'MAM', 'JJA', 'SON', 'Year']]

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))
predictions = loaded_model.predict(input_data)
df['Predicted_Annual_Anomaly'] = predictions

# creating a function for Prediction

def annual_average_temp_anomaly_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] > 0):
      return f'The annual average temp is predicted to be {prediction[0]:.2f}, which means it is more warmer this year than the baseline-average '
    elif (prediction[0] == 0):
      return f'The annual average temp is predicted to be {prediction[0]:.2f} which means it the same this year as the baseline-average' 
    else: 
     return f'The annual average temp is predicted to be  {prediction[0]:.2f} which means it is more colder this year than average'
    
  
def main():
    
    
# giving a title

# Convert the image to a base64 string
    import base64
    from io import BytesIO
    
    def img_to_base64(img_path):
     with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

    # Local image path
    image_path = '/Users/abelmetek/Desktop/Capstone/global-warming-1494965_1280.jpg'  # Replace with your image path
    img_base64 = img_to_base64(image_path)

    # Add CSS to set the background image
    st.markdown(
     f"""
     <style>
        .stApp {{
        background-image: url('data:image/jpeg;base64,{img_base64}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        }}

       .custom-text-box {{
        background-color: rgba(0, 0, 0, 0.7);  /* Dark background */
        color: white;  /* White text */
        padding: 20px;
        border-radius: 10px;
        font-size: 18px;
       }}
        </style>
        """, 
     unsafe_allow_html=True
)

    
    
    # getting the input data from the user
with st.sidebar:
    st.header( 'All inputs for tempratures are in °C'
    )
    
    Jan_Anomaly = st.number_input('Input the average anomomaly for month January')
    Feb_Anomaly = st.number_input('Input the average anomomaly for month February')
    Mar_Anomaly = st.number_input('Input the average anomomaly for month March')
    Apr_Anomaly = st.number_input('Input the average anomomaly for month April')
    May_Anomaly = st.number_input('Input the average anomomaly for month May')
    June_Anomaly = st.number_input('Input the average anomomaly for month June')
    July_Anomaly = st.number_input('Input the average anomomaly for month July')
    Aug_Anomaly = st.number_input('Input the average anomomaly for month August')
    Sept_Anomaly = st.number_input('Input the average anomomaly for month September')
    Oct_Anomaly = st.number_input('Input the average anomomaly for month October')
    Nov_Anomaly = st.number_input('Input the average anomomaly for month November')
    Dec_Anomaly = st.number_input('Input the average anomomaly for month December')
    December_to_November_Anomaly = st.number_input('Input the average anomomaly from December of last year to November of this year')
    DJF = st.number_input('Input the average anomomaly from December of last year to February of this year(Winter Months)')
    MAM = st.number_input('Input the average anomomaly from March to May(Spring Months)')
    JJA = st.number_input('Input the average anomomaly from July to August (Summer Months)')
    SON = st.number_input('Input the average anomomaly from September to November (Fall Months)')
    #Year = st.text_input('Input the Year you want to predict')
    Year = st.slider("Select the Year", 1880, 2100)
     # Prediction Button
    if st.button('Predict Average Annual Temperature Anomaly'):
     if not all([Jan_Anomaly, Feb_Anomaly, Mar_Anomaly, Apr_Anomaly, May_Anomaly, June_Anomaly, July_Anomaly, Aug_Anomaly, Sept_Anomaly, Oct_Anomaly, Nov_Anomaly, Dec_Anomaly, December_to_November_Anomaly, DJF, MAM, JJA, SON, Year]):
         st.error("Please fill in all input fields.")
     else:
         Januray_December_Anomaly = annual_average_temp_anomaly_prediction([Jan_Anomaly, Feb_Anomaly, Mar_Anomaly, Apr_Anomaly, May_Anomaly, June_Anomaly, July_Anomaly, Aug_Anomaly, Sept_Anomaly, Oct_Anomaly, Nov_Anomaly, Dec_Anomaly, December_to_November_Anomaly, DJF, MAM, JJA, SON, Year])
         st.success(Januray_December_Anomaly)

st.title('Annual Mean Temperature Anomaly Prediction Web App')
    
  
    
    
# Interactive Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Year'], y=df['J-D'], mode='lines', name='Historical Annual Anomalies'))
fig.add_trace(go.Scatter(x=df['Year'], y=df['Predicted_Annual_Anomaly'], mode='lines', name='Predicted Annual Anomalies', line=dict(dash='dash')))
fig.update_layout(title='Historical vs Predicted Annual Average Temperature Anomalies', xaxis_title='Year', yaxis_title='Temperature Anomaly (°C)')
st.plotly_chart(fig)

    # About Section
st.expander("About this App").markdown("""
    This app predicts the annual average temperature anomaly based on monthly and seasonal temperature anomalies.
    The model is trained on historical temperature data.
    """)    
    
    
if __name__ == '__main__':
    main()
