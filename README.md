<img src="/static/icons/OurLogo.png" alt="Alt Text" width="350" height="200">
<br>

## Project Introduction

Welcome to ProHealth360! Our innovative platform is dedicated to transforming patient care through cutting-edge technology. We are on a mission to redefine healthcare by enabling the precise and swift identification of seven major diseases: COVID-19, Breast Cancer, Alzheimer's, Brain Tumor, Diabetes, Kidney, and Heart diseases.

At ProHealth360, we harness the formidable capabilities of machine learning to deliver fast and trustworthy diagnoses. This approach not only empowers medical professionals with accurate information but also significantly improves patient monitoring and treatment strategies. Join us in shaping the future of healthcare with ProHealth360!
<br>
<img src="rating.png" alt="Alt Text" width="40" height="30"> So, at the beginning, we would like to explain why we chose this name "ProHealth360" for our site.
"ProHealth360" combines the word "Pro," which is short for "professional" or "proactive," with "Health" and "360" to suggest a comprehensive and professional approach to healthcare.

- "Pro": Implies professionalism, expertise, and a forward-thinking approach to healthcare. It suggests that our platform offers high-quality and advanced healthcare solutions.

- "Health": Clearly indicates that the focus of the platform is on health and healthcare-related services.

- "360": is often used in branding to convey a comprehensive or all-encompassing approach, our web application could suggest that our platform offers a full and holistic range of disease prediction and diagnosis services. It implies that we cover all aspects of healthcare and provide a well-rounded solution. Additionally, "360" can also evoke a sense of completeness, accuracy, and thoroughness in the predictions and diagnoses our platform offers. Ultimately, the choice of "360" in the name adds a dynamic and comprehensive feel to our healthcare application. so by that, we chose this name.
  So, "ProHealth360" could signify a platform that offers expert, proactive, and comprehensive healthcare solutions.
  <br>

## Problem Definition

- This project addresses a critical issue—the lack of rapid and accurate disease detection, especially for individuals without immediate access to healthcare services. Our motivation behind choosing this project name stems from the urgent need to bridge this gap in healthcare accessibility.

- By offering a swifter and more efficient alternative to conventional diagnostic methods, our initiative aims to expedite the treatment process, ultimately saving lives and improving patient outcomes. With the capacity to identify a wide range of diseases, including COVID-19, Brain Tumors, Breast Cancer, Alzheimer's, Brain Tumor, Diabetes, Kidney, and Heart diseases. Our system underscores the potential to revolutionize patient care on a global scale.

- Our project incorporates advanced machine learning models to provide a comprehensive solution that streamlines the entire disease diagnosis process from start to finish. By leveraging the synergistic power of these models, we empower individuals and healthcare professionals alike with accurate and timely information, enabling early intervention and better-informed treatment decisions.

- Our project incorporates advanced machine learning models to provide a comprehensive solution that streamlines the entire disease diagnosis process from start to finish. By leveraging the synergistic power of these models, we empower individuals and healthcare professionals alike with accurate and timely information, enabling early intervention and better-informed treatment decisions.

## Project Plan

- Data Collection, Cleaning, and Preparation:
  <br>
  In this initial stage, we will collect diverse datasets from various sources, including Kaggle, Google, and medical and hospital records. The meticulous process of data cleaning and preparation will follow. This entails not only addressing anomalies and inconsistencies but also performing imputation and data transformation to ensure that the datasets are in optimal condition to eventually help our models predict with precision. Our objective is to create datasets that are not only clean but also enriched to support the development of highly accurate disease detection models. This phase serves as the solid foundation upon which our entire project rests, setting the stage for subsequent phases.
- Model Training :
  <br>
  The heart of our project resides in this pivotal phase, where we will meticulously train advanced disease detection models tailored to the unique characteristics of each ailment. Our approach varies depending on the disease:

  - COVID-19:
    For the prediction of COVID-19, we will employ a diverse set of models, each designed for different data modalities. These models include InceptionV3_Chest, InceptionV3_CT, RESNET Chest, RESNET_CT, VGG Chest, VGG_CT, Xception_chest, and Xception_CT. These models are specifically optimized to analyze chest X-rays and CT scans, crucial tools in the diagnosis of COVID-19.

  - Breast Cancer:
    For the prediction of breast cancer, we will employ Classification and Regression Trees (CART), Linear Support Vector Machines (SVM), Gaussian Naive Bayes (NB), and k-Nearest Neighbors (KNN). These models will undergo extensive training using our meticulously curated datasets.

  - Alzheimer's Disease:
    In the case of Alzheimer's disease, we will leverage Convolutional Neural Networks (CNN), DenseNet, InceptionV3, ResNet50, and VGG16. These deep learning architectures have proven to be effective in handling the complexities associated with Alzheimer's prediction.

  - Brain Tumor:
    Brain tumor detection will be primarily driven by Convolutional Neural Networks (CNN), which excel in image-based analysis and can accurately identify the presence of brain tumors.

  - Diabetes:
    For diabetes prediction, we will harness the power of Support Vector Machines (SVM), Decision Tree Classifier, and other suitable algorithms. These models will be trained rigorously to provide accurate predictions related to diabetes.

  - Heart Disease:
    Detecting heart disease demands a comprehensive approach. Our ensemble of models will include Logistic Regression, Support Vector Machines (SVM), Decision Tree Classifier, Random Forest Classifier, and K-nearest Neighbors Classifier. These models will be trained intensively to deliver precise results in heart disease diagnosis.

  - Kidney Disease:
    The prediction of kidney disease will involve a range of models, including RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, LogisticRegression, k-Nearest Neighbors (KNN), Gaussian Naive Bayes (NB), Support Vector Machines (SVM), and DecisionTreeClassifier. Each model will be fine-tuned to ensure accurate outcomes in kidney disease assessment.

  Throughout this phase, our focus remains on achieving the highest levels of accuracy and reliability in disease prediction. By tailoring our model selection to the specific characteristics of each ailment, we aim to provide a comprehensive and effective solution for healthcare professionals and individuals seeking timely and precise diagnoses.
  <br>

- Web Application Development:
  <br>
  To make our project accessible and user-friendly, we will craft an intuitive web application. This application will serve as the primary interface for users, facilitating data input and displaying results in a clear and comprehensible manner. The interface will encompass a Graphical User Interface (GUI) and multiple HTML pages, each tailored for the specific detection of one of the seven diseases. Our aim is to create an interactive platform that enhances the user experience and empowers individuals to make informed healthcare decisions.

- Backend Development:
  <br>
  Python, in conjunction with the Flask web framework, will be employed to build the essential backend of our project. This pivotal component will be responsible for managing user input, processing data, and seamlessly connecting with our trained models for predictions. It will efficiently relay the results back to the frontend, ensuring a smooth and responsive user experience. This backend infrastructure will be the backbone of our web application, facilitating the entire diagnostic process from start to finish.

Through these meticulously planned phases, our project will not only deliver accurate disease detection but also prioritize user-friendliness and accessibility, ultimately contributing to the improvement of healthcare outcomes and patient care.
<br>

## Project Goals

Our overarching goal is to create a comprehensive healthcare solution that addresses critical disease detection needs. These goals encompass the entire project, from data input to diagnosis:

1. Enable Accurate Disease Detection: Develop a robust and accurate disease detection system capable of identifying a wide range of diseases, including COVID-19, Breast Cancer, Alzheimer's, Brain Tumor, Diabetes, Pneumonia, Heart Disease, and Kidney Disease. Accuracy is our utmost priority to ensure early intervention and improved patient care.

2. End-to-End Solution: Build an end-to-end solution that seamlessly integrates data collection, preprocessing, model training, and real-time diagnosis. This holistic approach streamlines the entire disease detection process, ensuring that healthcare professionals and users have access to a complete and efficient system.

3. Leverage Advanced Technology: Utilize cutting-edge technology, including Convolutional Neural Networks (CNN) for image-based disease detection and classical machine learning algorithms such as Support Vector Machines (SVM), Decision Trees, Random Forests, Naive Bayes, and k-Nearest Neighbors (KNN) for other data modalities. By harnessing the power of these models, we aim to provide precise and reliable diagnoses.

4. User-Friendly Interface: Offer a user-friendly, web-based interface that facilitates effortless interaction with the system. Users should be able to input relevant data easily and receive clear, interpretable results. The interface will be designed to enhance user experience and accessibility, promoting informed healthcare decisions.

5. Enhance Healthcare Accessibility: Improve accessibility to healthcare services, especially for individuals without immediate access to healthcare facilities. Our project strives to make high-quality disease detection available to a broader audience, regardless of location or resources.

6. Empower Healthcare Professionals: Provide healthcare professionals with a valuable tool for faster and more accurate disease diagnosis. This tool aims to complement their expertise and decision-making, ultimately improving patient outcomes.

7. Research and Innovation: Foster research and innovation in the field of disease detection by contributing to the development and evaluation of advanced machine learning and deep learning techniques. We aim to stay at the forefront of technological advancements in healthcare.

8. Continuous Improvement: Commit to continuous improvement by incorporating user feedback, refining models, and expanding the range of diseases that our system can detect. Our project will remain adaptive and responsive to evolving healthcare needs.

By aligning our project with these comprehensive goals, we aspire to revolutionize disease detection, enhance patient care, and contribute to the advancement of healthcare technology.
<br>

## Some of pages of our website

<img src="/static/Screenshots/23.png" alt="Alt Text" width="700" height="300">
<img src="/static/Screenshots/6.png" alt="Alt Text" width="600" height="350">
<br>
<img src="/static/Screenshots/5.png" alt="Alt Text" width="600" height="300">
<br><img src="/static/Screenshots/4.png" alt="Alt Text" width="600" height="300">
<br><img src="/static/Screenshots/3.png" alt="Alt Text" width="600" height="300">
<br><img src="/static/Screenshots/27.png" alt="Alt Text" width="600" height="300">
<br><img src="/static/Screenshots/28.png" alt="Alt Text" width="600" height="300">
<br><img src="/static/Screenshots/29.png" alt="Alt Text" width="600" height="300">
<br><img src="/static/Screenshots/30.png" alt="Alt Text" width="600" height="300">
<br><img src="/static/Screenshots/31.png" alt="Alt Text" width="600" height="300">
<br><img src="/static/Screenshots/2.png" alt="Alt Text" width="600" height="300">
<img src="/static/Screenshots/8.png" alt="Alt Text" width="600" height="350">
<img src="/static/Screenshots/9.png" alt="Alt Text" width="600" height="250">
<img src="/static/Screenshots/10.png" alt="Alt Text" width="600" height="350"><img src="/static/Screenshots/11.png" alt="Alt Text" width="600" height="350"><img src="/static/Screenshots/12.png" alt="Alt Text" width="600" height="350"><img src="/static/Screenshots/13.png" alt="Alt Text" width="600" height="350"><img src="/static/Screenshots/14.png" alt="Alt Text" width="600" height="250"><img src="/static/Screenshots/15.png" alt="Alt Text" width="600" height="350"><img src="/static/Screenshots/16.png" alt="Alt Text" width="600" height="350"><img src="/static/Screenshots/17.png" alt="Alt Text" width="600" height="300">
<img src="/static/Screenshots/18.png" alt="Alt Text" width="600" height="300">
<img src="/static/Screenshots/19.png" alt="Alt Text" width="600" height="350">
<img src="/static/Screenshots/20.png" alt="Alt Text" width="600" height="350">
<img src="/static/Screenshots/21.png" alt="Alt Text" width="600" height="350">
<img src="/static/Screenshots/22.png" alt="Alt Text" width="600" height="300">
<img src="/static/Screenshots/24.png" alt="Alt Text" width="600" height="300">
<img src="/static/Screenshots/25.png" alt="Alt Text" width="600" height="300">
<img src="/static/Screenshots/26.png" alt="Alt Text" width="600" height="300">
