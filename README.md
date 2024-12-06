![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)
![Status](https://img.shields.io/badge/status-development-orange)
# Empowering Wind Power Forecasting and Wind Ramp Detection Through a Data Analytics Marketplace

<img src="img/schema_predico.png" alt="Image Alt Text" width="500"/>

#### How PREDICO Platform Operates in Practice
Refer to the following repository for detailed instructions on setting up and running the PREDICO platform in practice : https://github.com/INESCTEC/predico-collabforecast



<div style="text-align: center;">
    <h1>Collaborative Forecasting Engine</h1>
</div>

<img src="img/PREDICO.png" alt="Image Alt Text" width="1000"/>

<div style="text-align: center;">
    <h2>Contribution Assessment Module</h2>
</div>


**PREDICO** utilizes the following methods to assess forecasters’ contributions:

* **Bootstrapped p-values** (applied in-sample)
* **Shapley values or permutation importance** (applied out-of-sample)

These methods are implemented to address two critical objectives:

* **Enhancing user trust**: Encouraging forecaster engagement.
* **Model debugging and refinement**: Interpreting and improving the forecasts combination mechanism.

It’s also important to remember the following:

* #### A forecaster considered to have low importance in a poorly performing model might be crucial for a high-performing model. </span>
* #### Contribution score doesn't indicate the intrinsic predictive value of a forecaster on its own but rather how significant that forecaster is to a specific model. </span>

<div style="text-align: center;">
    <img src="img/forecasters_contribution.png" alt="Image Alt Text" width="400"/>
</div>

##
### Permutation Importance
<div style="text-align: center;">
<img src="img/permutation.jpg" alt="Image Alt Text" width="700"/>
</div>

### Shapley Values Importance
<div style="text-align: center;">
<img src="img/shapley.jpg" alt="Image Alt Text" width="700"/>
</div>



### Contributions
* Giovanni Buroni giovanni.buroni@inesctec.pt
* Carla Gonçalves carla.s.goncalves@inesctec.pt
* José Andrade jose.r.andrade@inesctec.pt
* André Garcia andre.f.garcia@inesctec.pt
* Ricardo Bessa ricardo.j.bessa@inesctec.pt

### Contacts
If you have any questions regarding the methodology, please contact:
* Giovanni Buroni giovanni.buroni@inesctec.pt



