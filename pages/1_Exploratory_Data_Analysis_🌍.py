
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker


from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import plotly.express as px
import sweetviz as sv



st.set_page_config(page_title="Exploratory Data Analysis", page_icon="🌍")

st.title("Exploratory Data Analysis")

st.markdown(
    """

<h2>Data Cleaning</h2>

<p>Basic data cleaning processes were conducted before developing any further functionalities. After data loading, incorporate with the codebook, Variables 'IDATE','SEQNO', '_PSU', '_STSTR', '_STRWT', '_RAWRAKE', '_WT2RAKE', '_CLLCPWT', '_DUALCOR', '_LLCPWT2', '_LLCPWT' were all with value HIDDEN, and were dropped in the dataset. </p>


<h2>For the missing data</h2>

<p>1. Within the data, Variables with value NAN were encoded as -1 for convenient.  </p>

<p>2. Almost every variable derived from the BRFSS interview has a code category labeled refused and assigned values of 9, 99, or 999. These values may also be used to represent missing responses. Missing responses may be due to non-interviews (A non-interview response results when an interview ends prior to this question and an interviewer then codes the remaining responses as refused.) and missing responses due to skip patterns in the questionnaire. This code, however, may capture some questions that were supposed to have answers, but for some reason do not have them, and appeared as a blank or another symbol. Combining these types of responses into a single code requires vigilance on the part of data file users who wish to separate (1) results of respondents who did not receive a particular question and (2) results from respondents who, after receiving the question, gave an unclear answer or refused to answer it.  </p>

<h2>About data weighting</h2>

<p>The BRFSS is designed to obtain sample information on the adult US population residing in different states. BRFSS data weights incorporate the design of BRFSS survey and characteristics of the population to help make sample data more representative of the population from which the data were collected. BRFSS weighting methodology comprises 1) design factors or design weight, and 2) some form of demographic adjustment of the population—by iterative proportional fitting or raking.

<p>The design weight accounts for the probability of selection and adjusts for nonresponse bias and non- coverage errors. Design weights are calculated using the weight of each geographic stratum (_STRWT), the number of phones within a household (NUMPHON3), and the number of adults aged 18 years and older in the respondent’s household (NUMADULT). For cellphone respondents, both NUMPHON3 and NUMADULT are set to 1. The formula for the design weight is 
Design Weight = _STRWT * (1/NUMPHON3) * NUMADULT </p>

<p>The stratum weight (_STRWT) accounts for differences in the probability of selection among strata (subsets of area code or prefix combinations). It is the inverse of the sampling fraction of each stratum. There is rarely a complete correspondence between strata (which are defined by subsets of area code or prefix combinations) and regions—which are defined by the boundaries of government entities. 
BRFSS calculates the stratum weight (_STRWT) using the following items: </p>

<p>• Number of available records (NRECSTR) and the number of records users select (NRECSEL) within each geographic strata and density strata. </p>
<p>• Geographic strata (GEOSTR), which may be the entire state or a geographic subset (e.g., counties, census tracts). </p>
<p>• Density strata (_DENSTR) indicating the density of the phone numbers for a given block of numbers as listed or not listed. </p>
<p>For more information about the weighted data calculation, please read the 2022 BFRSS Codebook on the website https://www.cdc.gov/brfss/annual_data/annual_2022.html </p>



<h2>About this page</h2>

<p>There are four tabs within the exploratory data analysis: 'Data Overview', 'Data Visualizations', 'Profiling Report', and 'Sweetviz Report' (as shown in the sidebar). Please select a method of exploratory data analysis to delve into the data.</p>




""",

    unsafe_allow_html=True
)



# data_file = '/Users/xxxxxt/Desktop/brfss_cleaned.csv'
data_file = 'brfss_cleaned.csv'
data = pd.read_csv(data_file)


def eda(data):
    select_options = ["Please select an option...","Data Overview", "Data Visualization", "Profiling Report", "Sweetviz Report"]
    selected_option = st.sidebar.selectbox("Choose a way for data exploratory", select_options)
    
    
    if selected_option == "Please select an option...":
        st.write("Now let's begin our exploratory data analysis!🌍")

    elif selected_option == "Data Overview":

        st.markdown(
    """

<h2>Data Overview</h2>

""",

    unsafe_allow_html=True
)
        st.write("First 20 rows of the data:")
        st.write(data.head(20))

        st.write("Data Description:")
        st.write(data.describe())



    elif selected_option == "Data Visualization":
        st.header("Data Visualizations")
        plot_options = ["Please select a type of data visualization", "Bar plot", "Scatter plot", "Histogram", "Box plot"]
        selected_plot = st.selectbox("Please select a plot type", plot_options)

        if selected_plot == "Bar plot":
            st.subheader("Bar Plot Configuration")
            x_axis = st.selectbox("Select x-axis", data.columns)
            y_axis = st.selectbox("Select y-axis", data.columns)
            bar_fig = px.bar(data, x=x_axis, y=y_axis)
            st.plotly_chart(bar_fig)

        elif selected_plot == "Scatter plot":
            st.subheader("Scatter Plot Configuration")
            x_axis = st.selectbox("Select x-axis", data.columns)
            y_axis = st.selectbox("Select y-axis", data.columns)
            scatter_fig = px.scatter(data, x=x_axis, y=y_axis)
            st.plotly_chart(scatter_fig)

        elif selected_plot == "Histogram":
            st.subheader("Histogram Configuration")
            column = st.selectbox("Select a column", data.columns)
            bins = st.slider("Number of bins", 5, 100, 20)
            fig, ax = plt.subplots()
            sns.histplot(data[column], bins=bins, ax=ax)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right") 
            st.pyplot(fig)

        elif selected_plot == "Box plot":
            st.subheader("Box Plot Configuration")
            column = st.selectbox("Select a column", data.columns)
            fig, ax = plt.subplots()
            sns.boxplot(data[column], ax=ax)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right") 
            st.pyplot(fig)

    elif selected_option == "Profiling Report":
        selected_columns = st.multiselect("Select columns for profiling report", data.columns)
        if selected_columns:
            pr = ProfileReport(data[selected_columns], explorative=True)
            st.header('**Profiling Report**')
            st_profile_report(pr)
        else:
            st.warning("Please select at least one column for the profiling report.")

    elif selected_option == "Sweetviz Report":
        selected_columns = st.multiselect("Select columns for Sweetviz report", data.columns)
        if selected_columns:
            sweet_report = sv.analyze(data[selected_columns])
            st.header('**Sweetviz Report**')
            st.write(sweet_report.show_html())
        else:
            st.warning("Please select at least one column for the Sweetviz report.")

eda(data)


