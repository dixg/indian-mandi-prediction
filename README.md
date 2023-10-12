# Indian-mandi-prediction: Commodity Price Prediction for Indian Farmers

A web application to empower Indian farmers with predictive insights into agricultural commodity prices using advanced ML. Published in the International Journal of Multidisciplinary Research and Growth Evaluation,
[Article Link](https://www.researchgate.net/publication/363376035_Machine_Learning_Price_Prediction_of_Agricultural_Produces_for_Indian_Farmers_using_LSTM)

## Objective
- **Access:** Ensure farmers have access to real-time and historical commodity prices.
- **Predict:** Provide price predictions (daily, weekly, monthly) using LSTM ML models.

![Price Prediction](/images/cottonPricePredictionMonthly.png]
## Challenges
1. **Data Access:** Technological barriers prevent farmers from utilizing available governmental data.
2. **Price Fluctuations:** Lack of predictive insights leads to financial instability.

## Machine Learning Model
- **Model:** LSTM, chosen for its proficiency in time-series data and minimizing MSE.
- **Data Source:** The dataset was parsed from the [Indian National Agriculture Market website](https://enam.gov.in/)
- **Tech Stack:** Python, Keras, TensorFlow, Pandas, React, and Django.
  
## Web Development
- **Front-End**: Developed using React.
- **Back-End**: Utilized Django for server-side interactions.
- **Visualization**: Employed Matlab Plot to graphically represent the predicted results.

## Frameworks and Technologies
- **Python**: Development and training of LSTM model.
- **React**: Building the front end of the web app.
- **Django**: Backend development and server construction.
- **Keras**: Building and loading sequential LSTM models.
- **TensorFlow**: ML and AI model development.
- **Pandas**: Data manipulation and analysis.
- **Scikit-Learn**: Normalizing columns of pandas data frames for ML models.
- **NumPy**: Numerical computing.
- **Matplotlib**: Creating static, animated, and interactive visualizations in Python.

## Data Visualization
Utilizing Matplotlib, price data and predictions are presented in a user-friendly graphical format.

![Data Visualization](/images/cottonPricePredictionDaily.png)

## Getting Started with AgriPredict
1. Clone the [GitHub repository](#).
2. Install dependencies: Python, Keras, TensorFlow, Pandas, React, and Django.
3. Navigate to the directory and run the application.

## To run the project - 
In the project directory, you can run:
### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.\
You may also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can't go back!**

If you aren't satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you're on your own.

You don't have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn't feel obligated to use this feature. However we understand that this tool wouldn't be useful if you couldn't customize it when you are ready for it.

