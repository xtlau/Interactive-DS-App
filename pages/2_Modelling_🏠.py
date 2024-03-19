

import pandas as pd

from sklearn.model_selection import train_test_split

# Importing Algorithm for Simple Vector Machine
from sklearn.svm import SVC, SVR
# Importing Knn algorithm
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
# Importing  Decision Tree algorithm
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
# Importing Random Forest Classifer
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingRegressor,GradientBoostingClassifier,AdaBoostRegressor,AdaBoostClassifier
# Importing Naive Bayes algorithm
from sklearn.naive_bayes import GaussianNB
# Importing Linear and Logistic Regression
from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge, BayesianRidge, Lasso
# Importing accuracy score and mean_squared_error
from sklearn.metrics import mean_squared_error, accuracy_score,mean_absolute_error

import seaborn as sns

import streamlit as st

from sklearn.neural_network import MLPClassifier,MLPRegressor

st.title("Modelling")
# data_file = '/Users/xxxxxt/Desktop/brfss_cleaned.csv'
data_file = 'brfss_cleaned.csv'
data = pd.read_csv(data_file)


# Step1: Selecting algorithm
algorithm=st.selectbox("Select Supervised Machine Learning Algorithm",
                     ("KNN","SVM","Decision Tree","Naive Bayes","Random Forest","Linear Regression","Logistic Regression","Ridge","Gradient Boosting", "Bayesian Ridge", "Lasso","Neural Network")) 



# # Step2 : Selecting regressor or classifier

#Regressor and Classifier: KNN, SVM, Decision Tree, Random Forest
#Regressor: Linear Regression/Ridge/BayesianRidge/Lasso
#Classifier:Naive Bayes, Logistic Regression

# Step2: Selecting regressor or classifier
# Update to include models that are strictly regressors or classifiers, or can be both.
if algorithm in ['Linear Regression', 'Lasso', 'Ridge', 'Bayesian Ridge']:
    algorithm_type = "Regressor"
    st.sidebar.write(f"{algorithm} only does Regression")
elif algorithm in ['Naive Bayes', 'Logistic Regression']:
    algorithm_type = "Classifier"
    st.sidebar.write(f"{algorithm} only does Classification")
else:
    # For models that can be both, let the user decide
    algorithm_type = st.selectbox("Select Algorithm Type",
                                  ("Classifier", "Regressor"))



# Step3: Select X and Y
def input_output(data):
    selected_x_columns = st.multiselect("Select At Least One Feature (X)", data.columns)
    selected_y_column = st.selectbox("Select Target Variable (Y)", data.columns)
    X = data[selected_x_columns]
    Y = data[selected_y_column]

    return X, Y



# Step4-1: Adding Parameters For Classifier
def add_parameter_classifier_general(algorithm):

    # Declaring a dictionary for storing parameters
    params = dict()

    # Add paramters for SVM ---Checked----
    if algorithm == 'SVM':

        # Add regularization parameter from range 0.01 to 10.0
        c_regular = st.sidebar.slider('C (Regularization)', 0.01, 10.0)
        # Add kernel is the arguments in the ML model
        # Polynomial ,Linear, Sigmoid and Radial Basis Function are types of kernals 
        kernel_custom = st.sidebar.selectbox('Kernel', ('linear', 'poly ', 'rbf', 'sigmoid'))
        # Add parameters into dictionary
        params['C'] = c_regular
        params['kernel'] = kernel_custom

    # Adding Parameters for KNN ----Checked----
    elif algorithm == 'KNN':

        # Add Number of Neighbour (1-20) to KNN 
        k_n = st.sidebar.slider('Number of Neighbors (K)', 1, 20,key="k_n_slider")

        # Adding weights
        weights_custom = st.sidebar.selectbox('Weights', ('uniform', 'distance'))

        # Add parameters into dictionary
        params['K'] = k_n
        params['weights'] = weights_custom

    # Add Parameters for Naive Bayes ----Checked----
    # It doesn't have any paramter
    elif algorithm == 'Naive Bayes':
        st.sidebar.info("This is a simple algorithm. It doesn't have Parameters for hyperparameter tuning.")

    # Add Parameters for Decision Tree ----Checked----
    elif algorithm == 'Decision Tree':

        # Add max_depth
        max_depth = st.sidebar.slider('Max Depth', 2, 17)
        # Add criterion
        # mse is for regression (it is used in DecisionTreeRegressor)
	    # mse will give error in classifier so it is removed
        criterion = st.sidebar.selectbox('Criterion', ('gini', 'entropy'))

        # Add splitter
        splitter = st.sidebar.selectbox("Splitter", ("best", "random"))

        # Add to dictionary
        params['max_depth'] = max_depth
        params['criterion'] = criterion
        params['splitter'] = splitter

        # Exception Handling using try except block
        # Because we are sending this input in algorithm model it will show error before any input is entered
        # For this we will do a default random state till the user enters any state and after that it will be updated
        try:
            random = st.sidebar.text_input("Enter Random State")
            params['random_state'] = int(random)
        except:
            params['random_state'] = 4567

    # Add Parameters for Random Forest ----Checked----
    elif algorithm == 'Random Forest':

        # Add max_depth
        max_depth = st.sidebar.slider('Max Depth', 2, 17)

        # Add number of estimators
        n_estimators = st.sidebar.slider('Number of Estimators', 1, 90)

        # Add criterion
        # mse is for regression (it is used in RandomForestRegressor)
	    # mse will give error in classifier so it is removed
        criterion = st.sidebar.selectbox('Criterion', ('gini', 'entropy', 'log_loss'))


        # Add to dictionary
        params['max_depth'] = max_depth
        params['n_estimators'] = int(n_estimators)
        params['criterion'] = criterion

        # Exception Handling using try except block
        # Because we are sending this input in algorithm model it will show error before any input is entered
        # For this we will do a default random state till the user enters any state and after that it will be updated
        try:
            random = st.sidebar.text_input("Enter Random State")
            params['random_state'] = int(random)
        except:
            params['random_state'] = 4567

    # Adding Parameters for Logistic Regression ----Checked----
    elif algorithm == 'Logistic Regression':

        # Adding regularization parameter from range 0.01 to 10.0
        c_regular = st.sidebar.slider('C (Regularization)', 0.01, 10.0)
        params['C'] = c_regular
        # Taking fit_intercept
        fit_intercept = st.sidebar.selectbox("Fit Intercept", ('True', 'False'))
        params['fit_intercept'] = bool(fit_intercept)

        # Add Penalty 
        penalty = st.sidebar.selectbox("Penalty", ('l2', None))
        params['penalty'] = penalty

        # Add n_jobs
        n_jobs = st.sidebar.selectbox("Number of Jobs", (None, -1))
        params['n_jobs'] = n_jobs

        # Add solver
        solver = st.sidebar.selectbox("Solver", ('lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky','sag', 'saga'))
        params['solver'] = solver

    elif algorithm == 'Gradient Boosting':
        #Add loss
        loss = st.sidebar.selectbox("Loss",('log_loss','exponential'))
        params['loss'] = loss

        #Add n_estomators
        n_estimators = n_estimators = st.sidebar.slider('Number of Estimators', 1, 100)
        params['n_estimators'] = int(n_estimators)

        #Add learning rate
        learning_rate = st.sidebar.slider('Learning Rate', 0.01, 1.0)
        params['learning_rate'] = learning_rate


    elif algorithm == 'Neural Network':
        hidden_layer_sizes = st.sidebar.selectbox("Hidden Layer Sizes", ((100,), (50, 50), (100, 50)))
        params['hidden_layer_sizes'] = hidden_layer_sizes

        activation = st.sidebar.selectbox("Activation", ('logistic', 'tanh', 'relu'))
        params['activation'] = activation

        solver = st.sidebar.selectbox("Solver", ('lbfgs', 'sgd', 'adam'))
        params['solver'] = solver

        learning_rate_init = st.sidebar.slider('Learning Rate', 0.01, 1.0)
        params['learning_rate_init'] = learning_rate_init


    return params



# Step4-2: Adding Parameters for regressor
def add_parameter_regressor(algorithm):

    # Declaring a dictionary for storing parameters
    params = dict()

    # Deciding parameters based on algorithm
    # Add Parameters for Decision Tree  ----Checked----
    if algorithm == 'Decision Tree':

        # Add max_depth
        max_depth = st.sidebar.slider('Max Depth', 2, 17)

        # Add criterion
        # mse is for regression- It is used in DecisionTreeRegressor
        criterion = st.sidebar.selectbox('Criterion', ('absolute_error', 'squared_error', 'poisson', 'friedman_mse'))

        # Add splitter
        splitter = st.sidebar.selectbox("Splitter", ("best", "random"))

        # Adding to dictionary
        params['max_depth'] = max_depth
        params['criterion'] = criterion
        params['splitter'] = splitter

        # Exception Handling using try except block
        # Because we are sending this input in algorithm model it will show error before any input is entered
        # For this we will do a default random state till the user enters any state and after that it will be updated
        try:
            random = st.sidebar.text_input("Enter Random State")
            params['random_state'] = int(random)
        except:
            params['random_state'] = 4567

    # Adding Parameters for Linear Regression ----Checked----
    elif algorithm == 'Linear Regression':

        # Add fit_intercept
        fit_intercept = st.sidebar.selectbox("Fit Intercept", ('True', 'False'))
        params['fit_intercept'] = bool(fit_intercept)

        # Add n_jobs
        n_jobs = st.sidebar.selectbox("Number of Jobs", (None, -1))
        params['n_jobs'] = n_jobs

    # Add Parameters for Random Forest ----Checked----
    elif algorithm == 'Random Forest':

        # Add max_depth
        max_depth = st.sidebar.slider('Max Depth', 2, 17)

        # Add number of estimators
        n_estimators = st.sidebar.slider('Number of Estimators', 1, 90)

        # Add criterion
        # mse is for regression- It is used in RandomForestRegressor
        criterion = st.sidebar.selectbox('Criterion', ('absolute_error', 'squared_error', 'poisson', 'friedman_mse'))

        # Add to dictionary
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
        params['criterion'] = criterion

        # Exception Handling using try except block
        # Because we are sending this input in algorithm model it will show error before any input is entered
        # For this we will do a default random state till the user enters any state and after that it will be updated
        try:
            random = st.sidebar.text_input("Enter Random State")
            params['random_state'] = int(random)
        except:
            params['random_state'] = 4567

    # Add Parameter for Ridge ----Changed----
    elif algorithm == 'Ridge':

        #Add ridge_aplha
        alpha = st.sidebar.slider("Alpha", 0, 2)
        params['alpha'] = alpha

    elif algorithm == 'Gradient Boosting':
        #Add loss
        loss = st.sidebar.selectbox("Loss",('squared_error','absolute_error','huber','quantile'))
        params['loss'] = loss


        #Add learning rate
        learning_rate = st.sidebar.slider('Learning Rate', 0.01, 1.0)
        params['learning_rate'] = learning_rate


        #Add n_estomators
        n_estimators = n_estimators = st.sidebar.slider('Number of Estimators', 1, 100)
        params['n_estimators'] = int(n_estimators)


    elif algorithm == 'Bayesian Ridge':
        alpha_1 = st.sidebar.slider('Alpha 1', min_value=1e-6, max_value=1e-3, step=1e-5,format="%.1e")
        alpha_2 = st.sidebar.slider('Alpha 2', min_value=1e-6, max_value=1e-3, step=1e-5, format="%.1e")
        lambda_1 = st.sidebar.slider('Lambda 1', min_value=1e-6, max_value=1e-3, step=1e-5, format="%.1e")
        lambda_2 = st.sidebar.slider('Lambda 2', min_value=1e-6, max_value=1e-3, step=1e-5, format="%.1e")

        params['alpha_1'] = alpha_1
        params['alpha_2'] = alpha_2
        params['lambda_1'] = lambda_1
        params['lambda_2'] = lambda_2

    elif algorithm == 'Lasso':
        alpha = st.sidebar.slider('Alpha', 0.01,2.0)
        params['alpha'] = alpha


    elif algorithm == 'Neural Network':
        hidden_layer_sizes = st.sidebar.selectbox("Hidden Layer Sizes", ((100,), (50, 50), (100, 50)))
        params['hidden_layer_sizes'] = hidden_layer_sizes

        activation = st.sidebar.selectbox("Activation", ('logistic', 'tanh', 'relu'))
        params['activation'] = activation

        solver = st.sidebar.selectbox("Solver", ('lbfgs', 'sgd', 'adam'))
        params['solver'] = solver

        learning_rate_init = st.sidebar.slider('Learning Rate', 0.01, 1.0)
        params['learning_rate_init'] = learning_rate_init



    return params

#Step5
# Calling Function based on regressor and classifier
# Here since the parameters for regressor and classifier are same for some algorithm we can directly use this
# Because of this here except for this three algorithm we do not need to take parameters separately


if (algorithm_type == "Regressor") and (algorithm in ["Decision Tree","Random Forest", "Linear Regression", "Ridge", "Gradient Boosting",'Bayesian Ridge','Lasso','Neural Network']): ####----Changed----
    params = add_parameter_regressor(algorithm)
else :
    params = add_parameter_classifier_general(algorithm)


#Step6-1
# Now we will build ML Model for this dataset and calculate accuracy for that for classifier
def model_classifier(algorithm, params):

    if algorithm == 'KNN':
        return KNeighborsClassifier(n_neighbors=params['K'], weights=params['weights'])

    elif algorithm == 'SVM':
        return SVC(C=params['C'], kernel=params['kernel'])

    elif algorithm == 'Decision Tree':
        return DecisionTreeClassifier(
            criterion=params['criterion'], splitter=params['splitter'],
            random_state=params['random_state'])

    elif algorithm == 'Naive Bayes':
        return GaussianNB()

    elif algorithm == 'Random Forest':
        return RandomForestClassifier(n_estimators=params['n_estimators'],
                                      max_depth=params['max_depth'],
                                      criterion=params['criterion'],
                                      random_state=params['random_state'])


    elif algorithm == 'Logistic Regression':
        return LogisticRegression(fit_intercept=params['fit_intercept'],
                                  penalty=params['penalty'], C=params['C'], n_jobs=params['n_jobs'], solver=params['solver'])
    
    elif algorithm == 'Gradient Boosting':
        return GradientBoostingClassifier(loss = params['loss'],n_estimators=params['n_estimators'],learning_rate=params['learning_rate'])

    elif algorithm == 'Neural Network':
        return MLPClassifier(activation=params['activation'], solver=params['solver'], hidden_layer_sizes=params['hidden_layer_sizes'], learning_rate_init=params['learning_rate_init'])


#Step6-2
# Now we will build ML Model for this dataset and calculate accuracy for that for regressor
def model_regressor(algorithm, params):

    if algorithm == 'KNN':
        return KNeighborsRegressor(n_neighbors=params['K'], weights=params['weights'])

    elif algorithm == 'SVM':
        return SVR(C=params['C'], kernel=params['kernel'])

    elif algorithm == 'Decision Tree':
        return DecisionTreeRegressor(
            criterion=params['criterion'], splitter=params['splitter'],
            random_state=params['random_state'])

    elif algorithm == 'Random Forest':
        return RandomForestRegressor(n_estimators=params['n_estimators'],
                                      max_depth=params['max_depth'],
                                      criterion=params['criterion'],
                                      random_state=params['random_state'])

    elif algorithm == 'Linear Regression':
        return LinearRegression(fit_intercept=params['fit_intercept'], n_jobs=params['n_jobs'])
    
    elif algorithm == 'Ridge': ###----Added-----
        return Ridge(alpha=params['alpha'])
    
    elif algorithm == 'Gradient Boosting':
        return GradientBoostingRegressor(loss = params['loss'],n_estimators=params['n_estimators'],learning_rate=params['learning_rate'])
    
    elif algorithm == 'Bayesian Ridge':
        return BayesianRidge(alpha_1=params['alpha_1'], alpha_2=params['alpha_2'], lambda_1=params['lambda_1'], lambda_2=params['lambda_2'])
    
    elif algorithm == 'Lasso':
        return Lasso(alpha=params['alpha'])
   
    elif algorithm == 'Neural Network':
        return MLPRegressor(activation=params['activation'], solver=params['solver'], hidden_layer_sizes=params['hidden_layer_sizes'], learning_rate_init=params['learning_rate_init'])
    
    

if not data.empty: 
    X, Y = input_output(data)

    if len(X.columns) > 0 and Y is not None:
        
        # Now selecting classifier or regressor
        # Calling Function based on regressor and classifier
        if algorithm_type == "Regressor":
            algo_model = model_regressor(algorithm,params)
        else :
            algo_model = model_classifier(algorithm,params)

        # Now splitting into Testing and Training data
        
        split_size = st.slider('Data Split Ratio (% for Training Set)', 10, 90, 80, 5)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=split_size)

        # Training algorithm
        algo_model.fit(x_train,y_train)

        # Now we will find the predicted values
        predict=algo_model.predict(x_test)

        # Finding Accuracy
        # Evaluating/Testing the model
        if algorithm != 'Linear Regression' and algorithm_type != 'Regressor':
            # For all algorithm we will find accuracy
            st.write("Training Accuracy is:",algo_model.score(x_train,y_train)*100)
            st.write("Testing Accuracy is:",accuracy_score(y_test,predict)*100)
        else:
            # Checking for Error
            # Error is less as accuracy is more
            # For linear regression we will find error
            st.write("Mean Squared error is:",mean_squared_error(y_test,predict))
            st.write("Mean Absolute error is:",mean_absolute_error(y_test,predict))

            ##ADD PREDICTION
            ##MAY ADD VISUALIZATION FOR TUNING

                #Initialise the key in session state
        if 'clicked' not in st.session_state:
            st.session_state.clicked ={1:False}

        #Function to udpate the value in session state
        def clicked(button):
            st.session_state.clicked[button]= True

        st.button("Let's make predictions using this model", on_click = clicked, args=[1])

        if st.session_state.clicked[1]:

            # After training and evaluation, add user input functionality for prediction
            st.write("Input Data for Prediction")
            input_data = {}  # Dictionary to store user inputs
            for feature in X.columns:
                # You can customize the input method based on the type of data (numeric, categorical, etc.)
                # Here, we're assuming numeric inputs. For categorical data, consider using st.selectbox or similar.
                input_data[feature] = st.number_input(f"Input value for {feature}", format="%f")
            
            # Convert user inputs into a DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = algo_model.predict(input_df)
            
            # Display prediction
            st.write(f"Prediction: {prediction}")
            