# Conformal Locally Valid interval-Estimates for Regression

**Clover** is a Python package that introduces a novel class of methods for constructing local prediction intervals around regression point estimates, ensuring local coverage guarantees. Our methodology hinges on the creation of a feature space partition that closely approximates conditional coverage. This is achieved by fitting regression trees and Random Forests on conformity scores.

Our approach finds an optimal balance between theoretical robustness and practical efficiency. We construct prediction intervals that not only offer solid theoretical assurances, including marginal, local, and asymptotic conditional coverage, but also demonstrate superior scalability and performance when compared to conventional baselines. As a result, **clover** provides easily accessible adaptive prediction intervals, offering a more accurate representation of model uncertainty in regression problems.

All methods, experiments and properties are more detailed in the paper:

[Cabezas, L.M.C., Otto M.P., Izbicki R., Stern R.B. (2024). Regression Trees for Fast and Adaptive Prediction Intervals. arXiv preprint arXiv:2402.07357](https://arxiv.org/abs/2402.07357)

## Instalation

Clone the repo and run the following command in the clover directory to install clover


```python
pip install .
```

## Tutorial

We propose mainly two new methods to produce prediction intervals in regression problems: **LOCART** and **LOFOREST**.

* **LOCART**: Local Regression Trees (LOCART) is based on creating a partition $\mathcal{A}$ of the feature space, and defining the cutoffs by separately applying conformal prediction to each partition element $A \in \mathcal{A}$. The selection of $\mathcal{A}$ is guided by a data- driven optimization process designed to yield the most accurate estimates of the cutoffs.

* **LOFOREST**: Local Coverage Regression Forests (LOFOREST) builds on Locart by using multiple regression trees on conformity scores to define its prediction interval. That is, LOFOREST is a Random Forest of trees on conformity scores.

We can enhance both methods by strategically incorporating additional features into our feature matrix during the calibration phase. This includes A-LOCART and A-LOFOREST, which can utilize variance (difficulty) estimates, random projections, and other dispersion measures. Importantly, we can circumvent the limitations of dispersion measures by leveraging the feature selection capabilities of both CART and RF algorithms.

In the following tutorial, we show how to properly use our two primary methods, **LOCART** and **LOFOREST**, and, how to use their augmented variants as a bonus.

* **Simulating a heteroscedastic scenario to display the methods usage:**


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def make_variable_data(n, std_dev=1/5):
    x = np.random.uniform(low=-1, high=1, size=n)
    y = (x**3) + 2 * np.exp(-6 * (x - 0.3)**2)
    y = y + np.random.normal(scale=std_dev * np.abs(x), size=n)
    df = pd.DataFrame({'x': x, 'y': y})
    return df

# Set the seaborn style to white
sns.set_style("white")

# Generate the data
df = make_variable_data(1000)

# Create the scatterplot
sns.scatterplot(x='x', y='y', data=df, alpha = 1/5, color = 'black')
plt.show()
```


    
![png](README_files/README_8_0.png)
    


* **Let's prepare our training, calibration, and testing datasets. We will be using K-Nearest Neighbors (KNN) as our base model for regression.**


```python
from sklearn.neighbors import KNeighborsRegressor

# Set the random seed
np.random.seed(1500)

# Generate the training, calibration and validation sets
train_df = make_variable_data(1500)
cal_df = make_variable_data(1500)
val_df = make_variable_data(1500) 

# Create the KNN regressor
knn = KNeighborsRegressor(n_neighbors = 50)

# Fit the KNN regressor to the training data
knn.fit(train_df[['x']].values, train_df['y'].values)
```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KNeighborsRegressor(n_neighbors=50)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">KNeighborsRegressor</label><div class="sk-toggleable__content"><pre>KNeighborsRegressor(n_neighbors=50)</pre></div></div></div></div></div>



* **To compute predictive intervals using LOCART, we first need to define the *LocartSplit* class. This class primarily requires three initial arguments: the chosen conformity score, the base model, and the miscalibration level $\alpha$. The conformity score class can be directly imported from the Clover package, or alternatively, it can be created using the basic *Scores* class in a straightforward manner. In this example, we use the RegressionScore, which represents the mean absolute error. Additionally, we use the optional parameter *is_fitted* to indicate whether the base model has already been fitted. This prevents unnecessary re-training of the model.**


```python
from clover import LocartSplit, RegressionScore
# Set miscalibration level
alpha = 0.1

# Defining the class with conformity score, base model, miscalibration level and is_fitted paramter (optional)
locart = LocartSplit(RegressionScore, knn, alpha = alpha, is_fitted = True)
# fitting the base model to the training set
locart.fit(train_df[['x']].values, train_df['y'].values)

# computing local cutoffs by fitting regression tree to our calibration set
locart.calib(cal_df[['x']].values, cal_df['y'].values)
```




    {1: 0.2754178548446091,
     4: 0.1496540995480944,
     7: 0.06575805778537414,
     8: 0.06196180734978778,
     9: 0.13981516347329614,
     10: 0.26015728856970477}



* **To compute predictive intervals using LOFOREST, we redefine the *LocartSplit* class, this time changing the default value of the *cart_type* parameter to "RF". This parameter determines the method we use in the calibration step (calib), whether it's a single decision tree or a random forest.**


```python
# changing cart_type to "RF" to fit loforest
loforest = LocartSplit(RegressionScore, knn, alpha = alpha, cart_type = "RF", is_fitted = True)

# fitting base model
loforest.fit(train_df[['x']].values, train_df['y'].values)

# computing local cutoffs by fitting random forest to our calibration set
loforest.calib(cal_df[['x']].values, cal_df['y'].values)
```




    [{1: 0.272789376898864,
      4: 0.13357887384392392,
      5: 0.08496169979992507,
      6: 0.24521651783695064},
     {1: 0.2562442629456541,
      4: 0.11421809547184934,
      5: 0.09797065941092525,
      6: 0.24797030851503746},
     {1: 0.272789376898864,
      4: 0.1335832666375209,
      5: 0.0952247826851206,
      6: 0.24797030851503746},
     {2: 0.27528113566405926,
      4: 0.13996859735024125,
      5: 0.0679363429288516,
      6: 0.24403376359903875},
     {2: 0.26326350475116034,
      4: 0.11713609007705242,
      5: 0.11285310026890699,
      6: 0.26045917604835456},
     {1: 0.26327733941280396,
      4: 0.12857057816009132,
      5: 0.08440938639517792,
      6: 0.24521651783695064},
     {1: 0.27909722014059823,
      4: 0.14835222664332703,
      5: 0.08637904329809809,
      6: 0.24797030851503746},
     {1: 0.272789376898864,
      4: 0.13361606616304458,
      5: 0.0891172100356047,
      6: 0.24797030851503746},
     {1: 0.27528113566405926,
      4: 0.14513373847217084,
      5: 0.08637904329809809,
      6: 0.24797030851503746},
     {2: 0.25624732438348524,
      4: 0.12818692334972606,
      5: 0.09997130697242254,
      6: 0.2601581649290546},
     {1: 0.272789376898864,
      4: 0.10782180303133548,
      5: 0.12922926810502344,
      6: 0.26045917604835456},
     {2: 0.2562442629456541,
      4: 0.09020906343258515,
      5: 0.12922953251354605,
      6: 0.26015728856970477},
     {1: 0.263282041834742,
      4: 0.1285686540600608,
      5: 0.09797154289049814,
      6: 0.2568932760418084},
     {1: 0.2562442629456541,
      4: 0.11549555023438125,
      5: 0.09560995485080301,
      6: 0.24797030851503746},
     {1: 0.272789376898864,
      4: 0.14000205482155237,
      5: 0.09841993521720395,
      6: 0.26015771426921647},
     {2: 0.2562442629456541,
      4: 0.08698263240867983,
      5: 0.12887743395237494,
      6: 0.2604588549955983},
     {1: 0.272789376898864,
      4: 0.13262628629183365,
      5: 0.08440917947596205,
      6: 0.24521651783695064},
     {2: 0.272789376898864,
      4: 0.1335832666375209,
      5: 0.10077129443610854,
      6: 0.26015742246265466},
     {1: 0.272789376898864,
      4: 0.13361606616304458,
      5: 0.09957526229449598,
      6: 0.26015771426921647},
     {2: 0.26326350475116034,
      4: 0.1285657380326853,
      5: 0.1007697453881417,
      6: 0.26015771426921647},
     {2: 0.2562442629456541,
      4: 0.11704993622003838,
      5: 0.09560430639693829,
      6: 0.24797030851503746},
     {2: 0.272789376898864,
      4: 0.13262598476183726,
      5: 0.11286565409413858,
      6: 0.26015728856970477},
     {1: 0.2562442629456541,
      4: 0.11422138064369472,
      5: 0.09714091054021928,
      6: 0.24797030851503746},
     {2: 0.272789376898864,
      4: 0.13262598476183726,
      5: 0.1128671889891154,
      6: 0.2601571372880601},
     {2: 0.272789376898864,
      4: 0.13090116275644256,
      5: 0.09522493054198865,
      6: 0.24797030851503746},
     {1: 0.27279279044779153,
      4: 0.1336015520872901,
      5: 0.06895989726398669,
      6: 0.24403422880106476},
     {2: 0.272789376898864,
      4: 0.13357887384392392,
      5: 0.09522488599455459,
      6: 0.24797030851503746},
     {1: 0.272789376898864,
      4: 0.14515064788529825,
      5: 0.08147187741541045,
      6: 0.24521651783695064},
     {1: 0.27528910873675394,
      4: 0.13999760649866216,
      5: 0.08440897442088324,
      6: 0.2440361334961522},
     {2: 0.263282041834742,
      4: 0.12818805639097813,
      5: 0.11284891566049651,
      6: 0.26015742246265466},
     {1: 0.2754178548446091,
      4: 0.14925812903598268,
      5: 0.08440822164393284,
      6: 0.24521651783695064},
     {1: 0.26326350475116034,
      4: 0.11713609007705242,
      5: 0.10467586145441926,
      6: 0.2601581649290546},
     {2: 0.263282041834742,
      4: 0.1285686540600608,
      5: 0.1053227218836652,
      6: 0.2601571372880601},
     {1: 0.272789376898864,
      4: 0.13089869633963672,
      5: 0.08496150443818416,
      6: 0.2440361334961522},
     {2: 0.26326350475116034,
      4: 0.11711882220670941,
      5: 0.10467768880500278,
      6: 0.26015771426921647},
     {1: 0.2752995761785597,
      4: 0.1483517641194226,
      5: 0.10467604748374698,
      6: 0.26015759673601796},
     {1: 0.272789376898864,
      4: 0.13089748918493246,
      5: 0.09797123484540812,
      6: 0.2568932760418084},
     {1: 0.272789376898864,
      4: 0.13262539520318758,
      5: 0.0725121426593888,
      6: 0.24403422880106476},
     {1: 0.2727939522521634,
      4: 0.1399765253162693,
      5: 0.0814700722431135,
      6: 0.24521651783695064},
     {1: 0.27279512635076136,
      4: 0.13262598476183726,
      5: 0.10077129443610854,
      6: 0.26015742246265466},
     {1: 0.263282041834742,
      4: 0.1285686540600608,
      5: 0.08613352351630858,
      6: 0.24521651783695064},
     {2: 0.272789376898864,
      4: 0.10782180303133548,
      5: 0.12973966281385185,
      6: 0.2601571372880601},
     {1: 0.27278715925313224,
      4: 0.1300159042971965,
      5: 0.08440938639517792,
      6: 0.24521651783695064},
     {1: 0.2562442629456541,
      4: 0.11549483565607102,
      5: 0.09797209541581833,
      6: 0.2479839752893008},
     {2: 0.272789376898864,
      4: 0.13090116275644256,
      5: 0.09522493054198865,
      6: 0.24797030851503746},
     {1: 0.2754178548446091,
      4: 0.14965404445212033,
      5: 0.08682492718957537,
      6: 0.24797030851503746},
     {1: 0.2562442629456541,
      4: 0.11704993622003838,
      5: 0.06895054933529485,
      6: 0.2440333026649213},
     {1: 0.27278498640832427,
      4: 0.13001540191296315,
      5: 0.07251364978431614,
      6: 0.24403422880106476},
     {1: 0.26327733941280396,
      4: 0.1300117104809876,
      5: 0.10076820839502285,
      6: 0.26015742246265466},
     {1: 0.27528113566405926,
      4: 0.1300167081119699,
      5: 0.06793759200291583,
      6: 0.24350249427826015},
     {1: 0.272789376898864,
      4: 0.1308951251736366,
      5: 0.09560430639693829,
      6: 0.24797030851503746},
     {1: 0.272789376898864,
      4: 0.1335832666375209,
      5: 0.0952247826851206,
      6: 0.24797030851503746},
     {2: 0.263282041834742,
      4: 0.13088484002931552,
      5: 0.10467730247167333,
      6: 0.26045917604835456},
     {1: 0.25624732438348524,
      4: 0.11549822574851962,
      5: 0.08613352351630858,
      6: 0.24521651783695064},
     {1: 0.27279512635076136,
      4: 0.13262598476183726,
      5: 0.10467730247167333,
      6: 0.26045917604835456},
     {2: 0.2562442629456541,
      4: 0.11711038449733736,
      5: 0.07251785562801706,
      6: 0.24403422880106476},
     {2: 0.2562442629456541,
      4: 0.11709241149038188,
      5: 0.09522493054198865,
      6: 0.24797030851503746},
     {1: 0.263282041834742,
      4: 0.12818805639097813,
      5: 0.08452343322182437,
      6: 0.24403422880106476},
     {1: 0.272789376898864,
      4: 0.13262659246075306,
      5: 0.08911588117987053,
      6: 0.24797030851503746},
     {1: 0.272789376898864,
      4: 0.13361606616304458,
      5: 0.09957526229449598,
      6: 0.26015771426921647},
     {1: 0.272789376898864,
      4: 0.11237988001422723,
      5: 0.12048139825404519,
      6: 0.26015742246265466},
     {1: 0.263282041834742,
      4: 0.0961705788614724,
      5: 0.12871818918653186,
      6: 0.26015742246265466},
     {2: 0.263282041834742,
      4: 0.1308932807252629,
      5: 0.0995762235264756,
      6: 0.26015742246265466},
     {1: 0.2562442629456541,
      4: 0.11709241149038188,
      5: 0.08496188598975325,
      6: 0.24521651783695064},
     {1: 0.272789376898864,
      4: 0.1335832666375209,
      5: 0.09560662977426466,
      6: 0.24797481714159864},
     {1: 0.272789376898864,
      4: 0.1308951251736366,
      5: 0.10616451444639068,
      6: 0.26015742246265466},
     {1: 0.272789376898864,
      4: 0.10782180303133548,
      5: 0.12922926810502344,
      6: 0.26045917604835456},
     {1: 0.2752911564658066,
      4: 0.13999036967485562,
      5: 0.08440897442088324,
      6: 0.24403662095366271},
     {2: 0.24215917858868963,
      4: 0.0687469734265438,
      5: 0.11851200608428462,
      6: 0.2479839752893008},
     {2: 0.272789376898864,
      4: 0.1336015520872901,
      5: 0.10467596181234605,
      6: 0.26045917604835456},
     {2: 0.24025195305285765, 3: 0.09957671812982782, 4: 0.26015727906057284},
     {2: 0.272789376898864,
      4: 0.13361606616304458,
      5: 0.08440908603953451,
      6: 0.24521651783695064},
     {1: 0.272789376898864,
      4: 0.13999323832573388,
      5: 0.08146553435398099,
      6: 0.24403662095366271},
     {1: 0.272788262389727,
      4: 0.1308889502812551,
      5: 0.09714226625429176,
      6: 0.2568932760418084},
     {1: 0.27528113566405926,
      4: 0.1335922630788074,
      5: 0.08440889106618231,
      6: 0.24521651783695064},
     {1: 0.27528113566405926,
      4: 0.14513373847217084,
      5: 0.09797253743607447,
      6: 0.26015742246265466},
     {1: 0.25624732438348524,
      4: 0.0902087321426093,
      5: 0.1292297940036523,
      6: 0.26015698782908586},
     {2: 0.2562442629456541,
      4: 0.11704993622003838,
      5: 0.09797154289049814,
      6: 0.2568932760418084},
     {2: 0.24952955457441958,
      4: 0.06875373156257132,
      5: 0.12922966239843436,
      6: 0.26015728856970477},
     {2: 0.263282041834742,
      4: 0.09518220703403166,
      5: 0.12887736361246846,
      6: 0.26045917604835456},
     {1: 0.272789376898864,
      4: 0.14513642501444346,
      5: 0.08440863090878914,
      6: 0.24521651783695064},
     {1: 0.2562442629456541,
      4: 0.11549555023438125,
      5: 0.097971761815625,
      6: 0.2479839752893008},
     {1: 0.2752974361682349,
      4: 0.14835042095000417,
      5: 0.06793935379733833,
      6: 0.24403422880106476},
     {1: 0.272789376898864,
      4: 0.13361606616304458,
      5: 0.0676923158356902,
      6: 0.24350249427826015},
     {1: 0.26326350475116034,
      4: 0.11712738972699498,
      5: 0.09560362948933639,
      6: 0.24797030851503746},
     {2: 0.2562442629456541,
      4: 0.09020906343258515,
      5: 0.12887736361246846,
      6: 0.26045917604835456},
     {2: 0.2562442629456541,
      4: 0.11704993622003838,
      5: 0.08613352351630858,
      6: 0.24521651783695064},
     {1: 0.272789376898864,
      4: 0.1335832666375209,
      5: 0.10467730247167333,
      6: 0.26045917604835456},
     {1: 0.263282041834742,
      4: 0.13001364969478338,
      5: 0.08440938639517792,
      6: 0.24521651783695064},
     {1: 0.25624732438348524,
      4: 0.11549519033727611,
      5: 0.0861343926822125,
      6: 0.24521651783695064},
     {2: 0.2562442629456541,
      4: 0.09518042592098645,
      5: 0.12871855454928458,
      6: 0.26015728856970477},
     {1: 0.27279512635076136,
      4: 0.13998059361462578,
      5: 0.08682743213440769,
      6: 0.24797030851503746},
     {2: 0.26326350475116034,
      4: 0.11713609007705242,
      5: 0.10467768880500278,
      6: 0.26015742246265466},
     {1: 0.25624732438348524,
      4: 0.114225752994712,
      5: 0.09797209541581833,
      6: 0.2479839752893008},
     {2: 0.272789376898864,
      4: 0.13262539520318758,
      5: 0.09998090040962565,
      6: 0.26015771426921647},
     {2: 0.2562442629456541,
      4: 0.11549483565607102,
      5: 0.11284891566049651,
      6: 0.26015742246265466},
     {2: 0.272789376898864,
      4: 0.13089748918493246,
      5: 0.11678297244198446,
      6: 0.2601571372880601},
     {2: 0.26327733941280396,
      4: 0.13001291666706538,
      5: 0.0995762235264756,
      6: 0.26015771426921647},
     {1: 0.272789376898864,
      4: 0.13262539520318758,
      5: 0.09522439708483034,
      6: 0.24797030851503746},
     {1: 0.27528113566405926,
      4: 0.13262690337647745,
      5: 0.09957410387905344,
      6: 0.2601571372880601}]



* **Next, we will compare our suite of methods with the standard Regression Split. This comparison will help us evaluate the enhancements our methods bring to the table.**



```python
from mapie.regression import MapieRegressor

# fitting mapie
mapie = MapieRegressor(knn, method='base', cv='prefit')
mapie.fit(cal_df[['x']].values, cal_df['y'].values)

# values for prediction intervals
x_values = np.linspace(val_df['x'].min(), val_df['x'].max(), 500).reshape(-1, 1)
y_values = knn.predict(x_values)
y_pred, mapie_values = mapie.predict(x_values, alpha = alpha)
locart_values = locart.predict(x_values)
loforest_values = loforest.predict(x_values)


fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# plot regression split prediction intervals
sns.scatterplot(x='x', y='y', data=val_df, alpha=1/5, color = 'black', ax=axs[0]) 
axs[0].plot(x_values, y_values, color = 'tab:blue')
axs[0].plot(x_values, mapie_values[:,0], color = "tab:orange")
axs[0].plot(x_values, mapie_values[:,1], color = "tab:orange")
# axs[0].legend()  # Commented out to remove legend
axs[0].set_title('Regression Split')

# plot locart prediction intervals
sns.scatterplot(x='x', y='y', data=val_df, alpha=1/5, color = 'black', ax=axs[1]) 
axs[1].plot(x_values, y_values, color = 'tab:blue')
axs[1].plot(x_values, locart_values[:,0], color = "tab:green")
axs[1].plot(x_values, locart_values[:,1], color = "tab:green")
# axs[1].legend()  # Commented out to remove legend
axs[1].set_title('LOCART')

# plot locart prediction intervals
sns.scatterplot(x='x', y='y', data=val_df, alpha=1/5, color = 'black', ax=axs[2]) 
axs[2].plot(x_values, y_values, color = 'tab:blue')
axs[2].plot(x_values, loforest_values[:,0], color = "tab:red")
axs[2].plot(x_values, loforest_values[:,1], color = "tab:red")
# axs[2].legend()  # Commented out to remove legend
axs[2].set_title('LOFOREST')

plt.tight_layout()
plt.show()
```


    
![png](README_files/README_16_0.png)
    


* **We can also apply A-LOCART to the same problem simply by changing the *weighting* parameter from False to True.**


```python
# Changing the weighting parameter from False to True
a_locart = LocartSplit(RegressionScore, knn, alpha = alpha, is_fitted = True, weighting = True)

# fitting the base model to the training set
a_locart.fit(train_df[['x']].values, train_df['y'].values)

# computing local cutoffs by fitting regression tree to our calibration set
a_locart.calib(cal_df[['x']].values, cal_df['y'].values)

a_locart_values = a_locart.predict(x_values)

plt.figure(figsize = (10, 5))
sns.scatterplot(x='x', y='y', data=val_df, alpha=1/5, color = 'black') 
plt.plot(x_values, y_values, color = 'tab:blue')
plt.plot(x_values, a_locart_values[:,0], color = "tab:purple")
plt.plot(x_values, a_locart_values[:,1], color = "tab:purple")
plt.title('A-LOCART')
plt.show()
```


    
![png](README_files/README_18_0.png)
    


## Remark

While our framework is primarily designed to generate prediction intervals for regression problems, it's versatile enough to handle quantile regression and even conditional distributions. This is achieved by supplying the appropriate conformity score class and base model for quantile regression or conditional distribution. However, the application of **LOCART** and **LOFOREST** in these contexts is still under investigation by the authors.
