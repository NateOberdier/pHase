# pHase
Fermented Raw Milk Stage Prediction Model

### Problem Statement
Although raw milk fermentation follows a consistent pattern, there is currently no machine learning model that predicts the fermentation stage based on simple variables such as time, temperature, and culture addition due to lack of interest for scientists and lack of resources for consumers. Existing research largely covers pasteurized milk, commercial cultures, or industrial fermentation processes, leaving naturally fermenting raw milk minimally researched.

### Existing Solutions
Most existing studies focus on yogurt or kefir made using standardized starter cultures under strict laboratory conditions. These datasets do not reflect the microbial diversity or natural pH fluctuations seen in raw milk, and discontinue experimentation after the first few days. Spoilage prediction research also exists, but these models classify food as fresh or spoiled rather than identifying the current fermentation process. No accessible consumer oriented tools exist to estimate pH or the complete stage progression in raw milk.

### Proposed Solution
This project introduces pHase, a machine learning algorithm designed to model raw milk fermentation using an accurate 45 day dataset collected from 12 jars of raw milk. The project makes the following contributions:
1. It presents the first raw milk fermentation dataset collected at household room temperatures with both cultured and non-cultured jars.
2. It develops an incremental stage labeling method reflecting the natural occurring process of fermentation.
3. It trains two Random Forest models, one for pH regression and one for stage classification, that demonstrate high accuracy (RMSE ≈ 0.0375; stage accuracy 100%).
4. It introduces an interactive tool that allows users to input start time, temperature, and culture information to receive predictions about the current stage, pH, and upcoming transitions.

### Instructions
To run the project, go to pHase.ipynb. From here, you can open the code in Google Colab.
Press Ctrl+F9 on your keyboard or press "Runtime" and then "Run all".
All 4 code cells will be queued to execute and will run 1 at a time.
Scroll all the way to the bottom to use the project's main purpose, which is the Interactive raw-milk fermentation assistant.
You may now select a date, time, and if a culture has been added to your fermentation.
Now, press the "Predict" button to calculate the pH value, current stage, and future stages of your fermentation.

If unable to view in Colab, access the project through this link: https://colab.research.google.com/drive/1yys6XCieEJH8zbfMQggtwpWp8hl8YrTp?usp=sharing
