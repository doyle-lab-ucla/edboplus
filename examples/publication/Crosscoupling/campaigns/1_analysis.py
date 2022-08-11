
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import seaborn as sns

for campaign in ['challenging_campaign_cvt', 'challenging_campaign_random', 'easy_campaign']:

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))

    av_uncertainties_yield = []
    max_uncertainties_yield = []
    av_uncertainties_ee = []
    max_uncertainties_ee = []

    for round in range(1, 8):
        df = pd.read_csv(f"{campaign}/predictions_{round}.csv")


        max_uncertainties_yield.append(df['yield_predicted_variance'].max())
        max_uncertainties_ee.append(df['ee_predicted_variance'].max())

        av_uncertainties_yield.append(df['yield_predicted_variance'].mean())
        av_uncertainties_ee.append(df['ee_predicted_variance'].mean())

    max_uncertainties_yield = np.sqrt(max_uncertainties_yield)
    max_uncertainties_ee = np.sqrt(max_uncertainties_ee)
    av_uncertainties_yield = np.sqrt(av_uncertainties_yield)
    av_uncertainties_ee = np.sqrt(av_uncertainties_ee)
    plt.title(f"{campaign}", loc='center')
    sns.scatterplot(x=np.arange(1, 8), y=av_uncertainties_yield, ax=ax[0], label='average_uncertainty_yield')
    sns.scatterplot(x=np.arange(1, 8), y=av_uncertainties_ee, ax=ax[0], label='average_uncertainty_ee')
    plt.title(f"{campaign}", loc='center')
    sns.scatterplot(x=np.arange(1, 8), y=max_uncertainties_yield, ax=ax[1], label='max_uncertainty_yield')
    sns.scatterplot(x=np.arange(1, 8), y=max_uncertainties_ee, ax=ax[1], label='max_uncertainty_ee')

    ax[0].set_xlabel('Round')
    ax[0].set_ylabel('Uncertainty')
    ax[1].set_xlabel('Round')
    ax[1].set_ylabel('Uncertainty')
    ax[0].set_xticks(np.arange(1, 8))
    ax[1].set_xticks(np.arange(1, 8))
    ax[0].set_ylim(0, 15)
    ax[1].set_ylim(0, 15)

    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                 fancybox=True, shadow=True)
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                 fancybox=True, shadow=True)
    plt.tight_layout()


    # Expected improvement.
    av_eis_yield = []
    max_eis_yield = []
    av_eis_ee = []
    max_eis_ee = []

    for round in range(1, 8):
        df = pd.read_csv(f"{campaign}/predictions_{round}.csv")

        max_eis_yield.append(df['yield_expected_improvement'].max())

        max_eis_ee.append(df['ee_expected_improvement'].max())

        av_eis_yield.append(df['yield_expected_improvement'].mean())
        av_eis_ee.append(df['ee_expected_improvement'].mean())

    
    plt.title(f"{campaign}", loc='center')
    sns.scatterplot(x=np.arange(1, 8), y=av_eis_yield, ax=ax[2], label='average_EI_yield')
    sns.scatterplot(x=np.arange(1, 8), y=av_eis_ee, ax=ax[2], label='average_EI_ee')
    plt.title(f"{campaign}", loc='center')
    sns.scatterplot(x=np.arange(1, 8), y=max_eis_yield, ax=ax[3], label='max_EI_yield')
    sns.scatterplot(x=np.arange(1, 8), y=max_eis_ee, ax=ax[3], label='max_EI_ee')

    ax[2].set_xlabel('Round')
    ax[2].set_ylabel('EI')
    ax[3].set_xlabel('Round')
    ax[3].set_ylabel('EI')
    ax[2].set_xticks(np.arange(1, 8))
    ax[3].set_xticks(np.arange(1, 8))
    ax[2].set_ylim(0, 100)
    ax[3].set_ylim(0, 100)
    ax[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                 fancybox=True, shadow=True)
    ax[3].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                 fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(f"./plots/{campaign}.svg", format='svg')

    # Save results in csv file.
    df = pd.DataFrame([],
        columns=['max_uncertainty_yield', 'avg_uncertainty_yield', 'max_EI_yield', 'avg_EI_yield',
                 'max_uncertainty_ee', 'avg_uncertainty_ee', 'max_EI_ee', 'avg_EI_ee'])
    df['max_uncertainty_yield'] = max_uncertainties_yield
    df['max_uncertainty_ee'] = max_uncertainties_ee
    df['avg_uncertainty_yield'] = av_uncertainties_yield
    df['avg_uncertainty_yield'] = av_uncertainties_ee
    df['max_EI_yield'] = max_eis_yield
    df['max_EI_ee'] = max_eis_ee
    df['avg_EI_yield'] = av_eis_yield
    df['avg_EI_ee'] = av_eis_ee

    df.to_csv(f'crosscoupling_results_{campaign}.csv')
    plt.show()

