# 🧠 Social Media Listening for Mental Health Issues in SOCCSARGEN

A research project that uses Natural Language Processing (NLP) and a CNN-LSTM deep learning model to classify publicly available social media posts and comments related to mental health issues in SOCCSARGEN, Philippines. The project presents its findings through data visualizations, graphs, and heatmaps using Highcharts and a web application.

**Python NLP CNN-LSTM Power BI Data Visualization**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Features](#features)
- [Classification Categories](#classification-categories)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Methodology](#methodology)
- [System Architecture](#system-architecture)
- [Visualizations](#visualizations)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Ethical Considerations](#ethical-considerations)
- [Limitations](#limitations)
- [License](#license)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

## Overview

**Social Media Listening for Mental Health Issues in SOCCSARGEN** is an academic research project that explores the use of social media listening and deep learning to analyze publicly expressed mental health-related concerns in the SOCCSARGEN region of the Philippines, comprising South Cotabato, Sarangani, and General Santos City.

The study analyzes 5,067 manually collected public social media posts and comments from Facebook, X (formerly Twitter), TikTok, and Threads. Using NLP techniques and a CNN-LSTM model, the collected content is classified into predefined categories related to mental health.

The results are presented through data visualizations, including graphs and heatmaps, to help users explore the distribution of classified content.

### Problem Statement

Understanding mental health-related discussions on social media can be challenging because of the volume and variety of user-generated content.

- **Volume** — Manually examining large amounts of social media content can be time-consuming.
- **Text Complexity** — Social media posts often contain informal language, abbreviations, and varying expressions.
- **Classification** — Organizing posts into predefined mental health-related categories requires a consistent approach.
- **Data Interpretation** — Raw classification results can be difficult to interpret without effective visualization.

### Solution

This project applies NLP and a CNN-LSTM model to classify collected social media content into predefined categories. The results are then presented through Power BI visualizations and a web application, making the research findings easier to explore and interpret.

## Objectives

- Analyze publicly available social media posts and comments related to mental health in SOCCSARGEN.
- Apply NLP techniques to prepare textual data for classification.
- Develop and use a CNN-LSTM model for classifying content into predefined categories.
- Examine the distribution of classified posts and comments.
- Present research findings through graphs, heatmaps, and other data visualizations.

## Features

- 📱 **Multi-Platform Data Collection** — Uses manually collected public posts and comments from Facebook, X, TikTok, and Threads.
- 🧹 **Text Processing** — Applies NLP techniques to prepare text for analysis.
- 🧠 **Deep Learning Classification** — Uses a CNN-LSTM model to categorize social media content.
- 📊 **Category Distribution Analysis** — Presents the number and percentage of posts assigned to each category.
- 🗺️ **Data Visualization** — Uses graphs and heatmaps to present research findings.
- 🌐 **Web-Based Presentation** — Provides a web application for viewing and exploring the study's results.
- 📈 **Power BI Integration** — Uses Power BI to support data analysis and visualization.

## Classification Categories

The study classifies collected social media content into six categories.

| Category | Description |
|---|---|
| Stress | Posts classified as expressing or discussing stress-related concerns |
| Suicidal Behavior | Posts classified as related to suicidal behavior |
| Depression | Posts classified as related to depression |
| Anxiety | Posts classified as related to anxiety |
| Addiction | Posts classified as related to addiction |
| Trash / Irrelevant Content | Posts that do not fall within the study's relevant categories |

**Important:** These categories represent classifications of social media content. They do not establish the mental health status of the people who created the posts.

## Dataset

The study contains **5,067 manually collected public social media posts and comments** from four platforms.

| Platform | Data Collection Method |
|---|---|
| Facebook | Manual collection of publicly available posts and comments |
| X (formerly Twitter) | Manual collection of publicly available posts and comments |
| TikTok | Manual collection of publicly available posts and comments |
| Threads | Manual collection of publicly available posts and comments |

### Dataset Distribution

| Category | Number of Posts | Percentage |
|---|---:|---:|
| Stress | 1,569 | 30.96% |
| Suicidal Behavior | 947 | 18.69% |
| Depression | 894 | 17.64% |
| Anxiety | 787 | 15.53% |
| Addiction | 773 | 15.25% |
| Trash / Irrelevant Content | 97 | 1.91% |
| **Total** | **5,067** | **100%** |

### Key Observations

- Stress represents the largest category, accounting for 30.96% of the dataset.
- Suicidal behavior and depression are the second- and third-largest categories, respectively.
- Anxiety and addiction also account for substantial portions of the collected content.
- Trash or irrelevant content represents 1.91% of the dataset.

These figures describe the collected dataset, not the prevalence of mental health conditions among the general population of SOCCSARGEN.

## Tech Stack

| Component | Technology |
|---|---|
| Natural Language Processing | NLP |
| Deep Learning | CNN-LSTM |
| Data Collection | Manual collection of public social media posts and comments |
| Data Visualization | Highcharts.js |
| Frontend | HTML, JavaScript |
| Platform | Web Application |

## Features

- 📱 **Multi-Platform Data Collection** — Analyzes manually collected public posts and comments from Facebook, X (formerly Twitter), TikTok, and Threads.
- 🧹 **Text Processing** — Uses NLP techniques to prepare textual data for analysis.
- 🧠 **Deep Learning Classification** — Applies a CNN-LSTM model to classify content into predefined mental health-related categories.
- 📊 **Interactive Charts** — Uses Highcharts.js to present classification results through data visualizations.
- 🗺️ **Heatmap Visualization** — Displays relevant patterns in the analyzed data using heatmaps.
- 🌐 **Web-Based Dashboard** — Presents research findings through a web application built with HTML, JavaScript, and Highcharts.

## System Architecture

```text
┌──────────────────────────────────────┐
│         Social Media Platforms       │
│ Facebook | X | TikTok | Threads      │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│        Manual Data Collection        │
│       Public Posts and Comments      │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│       Text Processing Using NLP      │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│            CNN-LSTM Model            │
│       Mental Health Classification   │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│       Classification Results         │
│  Stress | Depression | Anxiety       │
│  Suicidal Behavior | Addiction       │
│       Trash / Irrelevant             │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│         Web-Based Visualization      │
│          HTML + JavaScript           │
│             Highcharts.js            │
└──────────────────────────────────────┘
```

## How It Works

### 1. Data Collection
Publicly available social media posts and comments are manually collected from Facebook, X, TikTok, and Threads.

### 2. Text Processing
NLP techniques are applied to prepare the collected textual data for classification.

### 3. Classification
A CNN-LSTM deep learning model classifies the prepared content into six predefined categories: stress, suicidal behavior, depression, anxiety, addiction, and trash or irrelevant content.

### 4. Data Analysis
The classification results are summarized to examine the distribution of content across the different categories.

### 5. Data Visualization
The processed results are presented through a web application using HTML, JavaScript, and Highcharts.js. Charts and heatmaps help users explore and interpret the research findings.

## Visualizations

The web application uses **Highcharts.js** to present the study's classification results through graphical visualizations.

- **Category Distribution** — Displays the number of posts assigned to each category.
- **Percentage Distribution** — Compares the proportion of content across categories.
- **Heatmaps** — Visualizes relevant patterns in the dataset.
- **Interactive Charts** — Enables users to explore the presented research results through the web interface.

📌 **Screenshots:** To be Added.
## Getting Started

The exact installation procedure depends on the source code and deployment environment used in the research. The following steps provide a general guide for accessing and reviewing the project.

### Prerequisites

- Access to the project repository.
- Access to the relevant source code and research files.
- Microsoft Power BI Desktop, if the report uses a `.pbix` file.
- The required programming language, libraries, and runtime environment used by the web application and classification model.

### Installation

**1. Clone the repository**

Replace `YOUR_USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR_USERNAME/SOCCSARGEN-Mental-Health-Listening.git
cd SOCCSARGEN-Mental-Health-Listening
```

**2. Open the project**

Open the repository in your preferred code editor or development environment.

**3. Configure the environment**

Install the dependencies and configure the application according to the project's actual implementation.

**4. Run the application**

Follow the setup instructions corresponding to the web framework, model, and visualization files included in the repository.

*Note: These are illustrative repository and setup names. Update them to match your actual GitHub repository and implementation.*

## Project Structure

The following is a suggested structure for organizing the research repository. Adjust it to reflect your actual files and directories.

```text
SOCCSARGEN-Mental-Health-Listening/
├── data/
│   └── README.md
├── notebooks/
│   └── model_development.ipynb
├── models/
│   └── README.md
├── powerbi/
│   └── README.md
├── web-app/
│   └── README.md
├── docs/
│   └── research-documentation.pdf
├── screenshots/
│   └── README.md
├── requirements.txt
├── LICENSE
└── README.md
```

**Privacy note:** Do not publish raw social media posts, usernames, identifiable information, model files, or research documents unless you have the appropriate permission and have reviewed the applicable privacy and research ethics requirements. The structure above is a template, not a claim that these files already exist.

## How It Works

### 1. Data Collection

Publicly available posts and comments are manually collected from the selected social media platforms. The study focuses on content relevant to mental health discussions within SOCCSARGEN.

### 2. Text Preparation

The collected content is organized and processed using NLP techniques to prepare it for classification. The exact preprocessing operations depend on the implemented workflow.

### 3. Text Classification

The CNN-LSTM model processes the prepared textual data and assigns it to one of the six predefined categories. The resulting labels are used for subsequent analysis.

### 4. Data Analysis

The classified results are summarized to determine the distribution of posts across categories. These summaries form the basis for the reported counts and percentages.

### 5. Visualization

The findings are presented through HeatMaps and a web application, using visual elements such as graphs and heatmaps to communicate the results.

## Ethical Considerations

This research uses manually collected, publicly available social media content. Public accessibility does not automatically remove privacy or ethical obligations.

- Personal and identifying information should be handled responsibly.
- Raw posts should not be republished unnecessarily.
- Research data should be stored and shared in accordance with applicable ethical requirements.
- Model classifications should not be treated as clinical diagnoses or confirmed indicators of an individual's mental health.
- Results should be interpreted within the limitations of the dataset and collection methodology.

## Limitations

- **Dataset Coverage** — The findings are based on the collected dataset and do not necessarily represent all social media users in SOCCSARGEN.
- **Platform Representation** — The collected data may not reflect the same level of activity or participation across platforms.
- **Language and Context** — Informal expressions, sarcasm, slang, and contextual meaning may affect classification.
- **Model Performance** — Classification errors may occur; performance should be assessed using documented evaluation metrics.
- **Interpretation** — A post's classification does not establish that its author has a particular mental health condition.

## Author

**Alexhzier Asdain**

B.S. Computer Science  
Notre Dame of Dadiangas University  
General Santos City, Philippines

## Acknowledgments

- **Notre Dame of Dadiangas University** — Academic guidance and research support.
- **Social Media Platforms** — Sources of publicly available content examined in the study.
- **Open-Source Community** — Developers and contributors whose tools and resources support NLP, deep learning, and data visualization research.

---

*This project demonstrates the application of social media listening, NLP, deep learning, and data visualization to the study of publicly expressed mental health-related concerns in SOCCSARGEN.*
