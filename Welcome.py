
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Interactive Data Science App for 2022 BRFSS Survey Data - Main Page"
)



st.sidebar.success("Please Select A Page Above.")

st.sidebar.success("Xiaotong Liu - University of Toronto")

st.markdown(
    """
<h1>Interactive Data Science App for 2022 BRFSS Survey Data - Main Page</h1>

<h2>About This App</h2>

<p>This interactive data science app utilizes BFRSS 2022 Survey Data. After a basic data cleaning process on data obtained from the Centers for Disease Control and Prevention Website (https://www.cdc.gov/brfss/annual_data/annual_2022.htm), it incorporates Exploratory Data Analysis, Machine Learning Modeling, and AI EDA Assistant functions. These functions enable users to conduct clinical data science exploration without coding themselves. Details for data cleaning will be included in the Exploratory Data Analysis Tab. Within each subpage, there will be introductions followed by instructions for that page.</p>

<h2>The Behavioral Risk Factor Surveillance System (BRFSS)</h2>


<p>The Behavioral Risk Factor Surveillance System (BRFSS) is a collaborative project between all the states in the United States, participating US territories, and the Centers for Disease Control and Prevention (CDC). Administered and supported by CDC's Population Health Surveillance Branch under the Division of Population Health at CDC’s National Center for Chronic Disease Prevention and Health Promotion, the BRFSS conducts ongoing health-related telephone surveys. These surveys collect data on health-related risk behaviors, chronic health conditions, health-care access, and use of preventive services from the noninstitutionalized adult population (≥ 18 years) residing in the United States and participating areas.</p>

<p>Initiated in 1984 with 15 states, the BRFSS has expanded over time and now collects data in all 50 states, the District of Columbia, and participating US territories. In 2022, all 50 states, the District of Columbia, Guam, Puerto Rico, and the US Virgin Islands participated in data collection.</p>

<p>The objective of the BRFSS is to gather uniform state-specific data on health risk behaviors, chronic diseases and conditions, access to health care, and use of preventive health services related to the leading causes of death and disability in the United States. The survey covers various factors such as health status and healthy days, exercise, cancer screenings, disability, oral health, chronic health conditions, tobacco use, alcohol consumption, and health-care access (core section). Optional Module topics for 2022 included social determinants of health and health equity, reactions to race, prediabetes and diabetes, cognitive decline, caregiver, cancer survivorship (type, treatment, pain management), and sexual orientation/gender identity (SOGI).</p>

<p>Since 2011, the BRFSS has conducted both landline telephone- and cellular telephone-based surveys, with all responses being self-reported; proxy interviews are not conducted. For the landline telephone survey, interviewers collect data from a randomly selected adult in a household, while for the cellular telephone survey, data is collected from adults answering cellular phones in private residences or college housing. Starting in 2014, all adults contacted through their cellular telephone became eligible for the survey, regardless of their landline phone use, resulting in complete overlap between the two survey methods.</p>

<p>State health departments oversee the field operations of the BRFSS, adhering to protocols established by their respective states and receiving technical support from the CDC. Collaboration occurs between state health departments during survey development, and they either conduct interviews themselves or enlist the help of contractors. After data collection, the information is sent to the CDC for editing, processing, weighting, and analysis. Each participating state health department receives an edited and weighted data file for each year of collection, with CDC also preparing summary reports of state-specific data. State health departments utilize BRFSS data for various purposes, such as identifying demographic disparities in health behaviors, designing and assessing public health programs, tackling emerging health concerns, proposing health-related legislation, and tracking progress toward state health objectives.</p>

""",

    unsafe_allow_html=True
)


