from scipy.stats.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

__all__ = [
    'Correlation', 'performance1', 'performance2', 'log_plot', 'make_log_plot'
]

class Correlation:

    r"""
    A correlation class for pearson and chatterjee method of statistical significance.

    Parameters
    ----------

    df : pd.DataFrame
        Takes in only the dataframe


    """
    def __init__(self, dataframe:pd.DataFrame):

        self._df = dataframe


    def _chatterjee(self, x:pd.Series, y:pd.Series) -> float:
        '''
        A private method that implements chatterjee method

        Return
        ------
        correlation between two variable
        '''
        df = pd.DataFrame()
        df['x_rk'] = x.rank()
        df['y_rk'] = y.rank()
        df = df.sort_values('x_rk')
        sum_term = df['y_rk'].diff().abs().sum()
        chatt_corr = (1 - 3 * sum_term / (pow(df.shape[0], 2) - 1))

        return chatt_corr

    def corr(self, method:str='chatterjee'):

        r'''

        Function to calculate the linear (Pearson's) and non-linear (Chatterjee's) relationships between log curves.
        Relationship between well logs are usually non-linear.

        Parameters
        ----------

        method : str, default 'chatterjee'
              Method of correlation. {'chatterjee', 'pearsonr', 'linear', 'nonlinear'}

              * 'linear' is the same as 'pearsonr'
              * 'nonlinear' is the same as 'chatterjee'

        Returns
        -------
        Correlation matrix of all possible log curves combination

        Example
        -------
         >>> corr = Correlation(df)
         >>> v = corr.corr(method='chatterjee)

        '''

        self._method = method
        X = self._df.columns.tolist()
        Y = X.copy()

        df = pd.DataFrame(index=X, columns=Y)

        for i in X:
            for j in Y:
                if method == 'chatterjee' or method == 'nonlinear':
                    corr = self._chatterjee(self._df[i], self._df[j])
                    df[i][j] = corr
                elif method=='pearsonr' or method == 'linear':
                    self._df = self._df.dropna()
                    corr, _ = pearsonr(self._df[i], self._df[j])
                    df[i][j] = corr

        #convert the columns to numeric from object
        for column in df.columns:

            df[column] = df[column].astype(np.float32)

        return df


    def plot_heatmap(self, title:str='Correlation Heatmap with Chatterjee', figsize:tuple=(12, 7), annot:bool=True, cmap=None):

        r'''
        Plots the heat map of Correlation Matrix

        Parameters
        ----------
        title : str
            Title of plot

        figsize : tuple
            Size of plot

        annot : bool, default True
            To annotate the coefficient in the plot

        cmap : matplotlib colormap name or object, or list of colors, optional
            The mapping from data values to color space

        Example
        -------
         >>> corr = Correlation(df)
         >>> v = corr.corr(method='chatterjee)
         >>> corr.plot_heatmap(cmap='Reds')

        '''

        corr = self.corr(self._method)
        mask_triangle=np.triu(np.ones(corr.shape)).astype(bool)
        plt.rcParams['figure.figsize'] = figsize
        plt.title(title)
        sns.heatmap(corr, mask=mask_triangle, annot=annot, cmap=cmap)

def make_log_plot(logs, x1, x2, x3, x4, x5, x6, x7, x8, well_name):

    """
    DESC: Displays log plot
    args::
        logs: dataframe
        x1: GR
        x2: RT
        x3: NPHI
        x4: RHOB
        x5: CALI
        x6: DTC
        x7: DTS
        well_name: str
    Returns plots

    """

    ztop = logs['DEPTH'].min(); zbot=logs['DEPTH'].max()

    #defining plot figure
    f, ax = plt.subplots(nrows=1, ncols=5, figsize=(12, 12), sharey=True)
    ax[0].set_ylabel("DEPTH(m)")
    f.suptitle(f'Well - {well_name}', fontsize=20, y=1.02)

    #for gamma ray track
    ax[0].minorticks_on()
    ax[0].grid(which='major', linestyle='-', linewidth=1, color='darkgrey')
    ax[0].yaxis.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
    ax[0].plot(logs[x1], logs['DEPTH'], 'black')
    ax[0].set_xlim(0, 200)#logs[x1].min(), logs[x1].max())
    ax[0].set_xlabel('GR(gAPI)', fontsize=12)
    ax[0].xaxis.label.set_color('black')
    ax[0].tick_params(axis='x', colors='black')
    ax[0].spines['top'].set_edgecolor('black')
    ax[0].spines["top"].set_position(("axes", 1.02))
    ax[0].set_ylim(ztop, zbot)
    ax[0].xaxis.set_ticks_position("top")
    ax[0].xaxis.set_label_position("top")
    ax[0].invert_yaxis()

    #for resitivity log
    ax[1].minorticks_on()
    ax[1].grid(which='major', linestyle='-', linewidth=1.0, color='darkgrey')
    ax[1].grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
    ax[1].plot(logs[x2], logs['DEPTH'], '--r')
    ax[1].set_xlim(0.2, 2000)
    ax[1].set_xlabel('RT(ohm-m)', fontsize=12)
    ax[1].xaxis.label.set_color('red')
    ax[1].tick_params(axis='x', colors='red')
    ax[1].spines['top'].set_edgecolor('red')
    ax[1].spines["top"].set_position(("axes", 1.02))
    ax[1].set_ylim(ztop, zbot)
    ax[1].xaxis.set_ticks_position("top")
    ax[1].xaxis.set_label_position("top")
    ax[1].invert_yaxis()
    ax[1].set_xscale("log")

    #for bulk density
    ax[2].minorticks_on()
    ax[2].yaxis.grid(which='major', linestyle='-', linewidth=1, color='darkgrey')
    ax[2].yaxis.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
    ax[2].set_xticklabels([]);ax[2].set_xticks([])
    rhob_ = ax[2].twiny()
    rhob_.plot(logs[x3], logs['DEPTH'], '-r', linewidth=1)
    rhob_.set_xlim(1.95, 2.95)
    rhob_.set_xlabel('RHOB(g/cm3)', fontsize=12)
    rhob_.xaxis.label.set_color('red')
    rhob_.tick_params(axis='x', colors='red')
    rhob_.spines['top'].set_edgecolor('red')
    rhob_.spines["top"].set_position(("axes", 1.02))
    rhob_.set_ylim(ztop, zbot)
    rhob_.xaxis.set_ticks_position("top")
    rhob_.xaxis.set_label_position("top")
    rhob_.invert_yaxis()

    #for neutron porosity
    nphi_ = ax[2].twiny()
    nphi_.grid(which='major', linestyle='-', linewidth=0.5, color='darkgrey')
    nphi_.plot(logs[x4], logs['DEPTH'], '--b', linewidth=1)
    nphi_.set_xlim(0.45, -0.15)
    nphi_.set_xlabel('NPHI(v/v)', fontsize=12)
    nphi_.xaxis.label.set_color('blue')
    nphi_.tick_params(axis='x', colors='blue')
    nphi_.spines['top'].set_edgecolor('blue')
    nphi_.spines["top"].set_position(("axes", 1.07))
    nphi_.set_ylim(ztop, zbot)
    nphi_.xaxis.set_ticks_position("top")
    nphi_.xaxis.set_label_position("top")
    nphi_.invert_yaxis()

    #for DTC
    ax[3].minorticks_on()
    ax[3].yaxis.grid(which='major', linestyle='-', linewidth=1, color='darkgrey')
    ax[3].yaxis.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
    ax[3].set_xticklabels([]);ax[3].set_xticks([])
    dtc_ = ax[3].twiny()
    dtc_.plot(logs[x5], logs['DEPTH'], '-', c='purple', linewidth=1)
    dtc_.set_xlim(40, 240)
    dtc_.set_xlabel('True DTC(us/m)', fontsize=12)
    dtc_.xaxis.label.set_color('purple')
    dtc_.tick_params(axis='x', colors='purple')
    dtc_.spines['top'].set_edgecolor('purple')
    dtc_.spines["top"].set_position(("axes", 1.02))
    dtc_.set_ylim(ztop, zbot)
    dtc_.xaxis.set_ticks_position("top")
    dtc_.xaxis.set_label_position("top")
    dtc_.invert_yaxis()

    #for predicted DTC
    dts_ = ax[3].twiny()
    dts_.grid(which='major', linestyle='-', linewidth=0.5, color='darkgrey')
    dts_.plot(logs[x6], logs['DEPTH'], '-', c='black', linewidth=1)
    dts_.set_xlim(40, 240)
    dts_.set_xlabel('Predicted DTC(us/m)', fontsize=12)
    dts_.xaxis.label.set_color('black')
    dts_.tick_params(axis='x', colors='black')
    dts_.spines['top'].set_edgecolor('black')
    dts_.spines["top"].set_position(("axes", 1.07))
    dts_.set_ylim(ztop, zbot)
    dts_.xaxis.set_ticks_position("top")
    dts_.xaxis.set_label_position("top")
    dts_.invert_yaxis()

    #for DTS
    ax[4].minorticks_on()
    ax[4].yaxis.grid(which='major', linestyle='-', linewidth=1, color='darkgrey')
    ax[4].yaxis.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
    ax[4].set_xticklabels([]);ax[4].set_xticks([])
    true = ax[4].twiny()
    true.plot(logs[x7], logs['DEPTH'], '-', c='purple', linewidth=1)
    true.set_xlim(80, 380)
    true.set_xlabel('True DTS(us/m)', fontsize=12)
    true.xaxis.label.set_color('purple')
    true.tick_params(axis='x', colors='purple')
    true.spines['top'].set_edgecolor('purple')
    true.spines["top"].set_position(("axes", 1.02))
    true.set_ylim(ztop, zbot)
    true.xaxis.set_ticks_position("top")
    true.xaxis.set_label_position("top")
    true.invert_yaxis()

    #for predicted DTS
    pred = ax[4].twiny()
    pred.grid(which='major', linestyle='-', linewidth=0.5, color='darkgrey')
    pred.plot(logs[x8], logs['DEPTH'], '-', c='black', linewidth=1)
    pred.set_xlim(80, 380)
    pred.set_xlabel('Predicted DTS(us/m)', fontsize=12)
    pred.xaxis.label.set_color('black')
    pred.tick_params(axis='x', colors='black')
    pred.spines['top'].set_edgecolor('black')
    pred.spines["top"].set_position(("axes", 1.07))
    pred.set_ylim(ztop, zbot)
    pred.xaxis.set_ticks_position("top")
    pred.xaxis.set_label_position("top")
    pred.invert_yaxis()

    plt.tight_layout(h_pad=1)
    f.subplots_adjust(wspace=0.0)
    
    plt.show()

def performance1(
        train_dtc_loss, train_dtc_score, train_dts_score,
        train_dts_loss, val_dtc_score, val_dtc_loss,
        val_dts_score, val_dts_loss
        ):
    # Creating subplots
    fig, axs = plt.subplots(2, 2, figsize=(30, 15))
    epochs_range = range(len(train_dtc_loss))

    # Plotting for subplot 1
    axs[0, 0].plot(epochs_range, train_dtc_score, color='tab:blue')
    axs[0, 0].set_ylabel('R2 Score', color='tab:blue')
    axs[0, 0].tick_params(axis='y', labelcolor='tab:blue')
    axs[0, 0].set_xlabel('No. of Iterations')

    axs2_1 = axs[0, 0].twinx()
    axs2_1.plot(epochs_range, train_dtc_loss, color='tab:red')
    axs2_1.set_ylabel('Root Mean Squared Error', color='tab:red')
    axs2_1.tick_params(axis='y', labelcolor='tab:red')
    axs2_1.set_xlabel('No of iterations')
    axs[0, 0].set_title('Train DTC Score and Loss')

    # Plotting for subplot 2
    axs[0, 1].plot(epochs_range, train_dts_score, color='tab:blue',)
    axs[0, 1].set_ylabel('R2 Score', color='tab:blue')
    axs[0, 1].tick_params(axis='y', labelcolor='tab:blue')
    axs[0, 1].set_xlabel('No. of Iterations')

    axs2_2 = axs[0, 1].twinx()
    axs2_2.plot(epochs_range, train_dts_loss, color='tab:red')
    axs2_2.set_ylabel('Root Mean Squared Error', color='tab:red')
    axs2_2.tick_params(axis='y', labelcolor='tab:red')
    axs2_2.set_xlabel('No of iterations')
    axs[0, 1].set_title('Train DTS Score and Loss')

    # Plotting for subplot 3
    axs[1, 0].plot(epochs_range, val_dtc_score, color='tab:blue')
    axs[1, 0].set_ylabel('R2 Score', color='tab:blue')
    axs[1, 0].tick_params(axis='y', labelcolor='tab:blue')
    axs[1, 0].set_xlabel('No. of Iterations')

    axs2_3 = axs[1, 0].twinx()
    axs2_3.plot(epochs_range, val_dtc_loss, color='tab:red')
    axs2_3.set_ylabel('Root Mean Squared Error', color='tab:red')
    axs2_3.tick_params(axis='y', labelcolor='tab:red')
    axs2_3.set_xlabel('No of iterations')
    axs[1, 0].set_title('Validation DTC Score and Loss')

    # Plotting for subplot 4
    axs[1, 1].plot(epochs_range, val_dts_score, color='tab:blue')
    axs[1, 1].set_ylabel('R2 Score', color='tab:blue')
    axs[1, 1].tick_params(axis='y', labelcolor='tab:blue')
    axs[1, 1].set_xlabel('No. of Iterations')

    axs2_4 = axs[1, 1].twinx()
    axs2_4.plot(epochs_range, val_dts_loss, color='tab:red')
    axs2_4.set_ylabel('Root Mean Squared Error', color='tab:red')
    axs2_4.tick_params(axis='y', labelcolor='tab:red')
    axs2_4.set_xlabel('No of iterations')
    axs[1, 1].set_title('Validation DTS Score and Loss')

    plt.show()

def performance2(
        train_dtc_loss, train_dtc_score, train_dts_score,
        train_dts_loss, val_dtc_score, val_dtc_loss,
        val_dts_score, val_dts_loss
        ):
    epochs_range = range(len(train_dtc_loss))

    plt.figure(figsize=(30, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_dtc_score, label='Training Accuracy (DTC)')
    plt.plot(epochs_range, val_dtc_score, label='Validation Accuracy (DTC)')
    plt.legend(loc='lower right')
    plt.xlabel('No of Iterations')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_dtc_loss, label='Training Loss (DTC)')
    plt.plot(epochs_range, val_dtc_loss, label='Validation Loss (DTC)')
    plt.legend(loc='upper right')
    plt.xlabel('No of Iterations')
    plt.ylabel('Root Mean Squared Error')
    plt.title('Training and Validation Loss')

    plt.show()

    plt.figure(figsize=(30, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_dts_score, label='Training Accuracy (DTS)')
    plt.plot(epochs_range, val_dts_score, label='Validation Accuracy (DTS)')
    plt.legend(loc='lower right')
    plt.xlabel('No of Iterations')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_dts_loss, label='Training Loss (DTS)')
    plt.plot(epochs_range, val_dts_loss, label='Validation Loss (DTS)')
    plt.legend(loc='upper right')
    plt.xlabel('No of Iterations')
    plt.ylabel('Root Mean Squared Error')
    plt.title('Training and Validation Loss')

    plt.show()


def log_plot(well, well_name):
    # Display the test data
    #well = well # test wells: well2, well5
    '''
    This is yohannes code that returns plot for test data
    args::
        well: dataframe
        well_name = well name in str
    
    '''

    # define what logs are we going to use
    logs = ['GR', 'RT', 'NPHI', 'RHOB', 'DTC', 'DTS']

    # titles to show
    title = ['GR', 'RT', 'NPHI', 'RHOB', 'DTC', 'DTS']

    # create the subplots; ncols equals the number of logs
    fig, ax = plt.subplots(nrows=1, ncols=len(logs), figsize=(15,10))
    fig.suptitle(f'Well {well_name}', size=20, y=1.05)

    # looping each log to display in the subplots

    colors = ['purple', 'purple', 'purple', 'purple', 'red', 'green']

    for i in range(len(logs)):
        if i == 1:
            # for resistivity, semilog plot
            ax[i].semilogx(well[logs[i]], well['DEPTH'], color=colors[i])
        else:
            # for non-resistivity, normal plot
            ax[i].plot(well[logs[i]], well['DEPTH'], color=colors[i])

        ax[i].set_ylim(max(well['DEPTH']), min(well['DEPTH']))
        ax[i].set_title(title[i], pad=15)
        ax[i].grid(True)

#     ax[2].set_xlim(0, 200)
    plt.tight_layout(1)
    plt.show()
