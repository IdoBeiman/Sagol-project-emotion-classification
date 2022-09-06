import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_predictions(predictions_path, results_dir):

    results = pd.read_csv(predictions_path)
    results.drop(['Unnamed: 0'], axis=1, inplace=True)
    iter_num = int(len(results.columns) / 6)
    df = results.iloc[:, 0:6]
    df.columns = [col.split('_')[0] for col in df.columns]
    for i in range(1, iter_num):
        new_df = results.iloc[:, 0+6*i:6+6*i]
        new_df.columns = [col.split('_')[0] for col in new_df.columns]
        df = pd.concat([df, new_df], ignore_index=True)

    models = [m for m in df.columns if m not in ["BL", "Real"]]

    y_val = df['Real']
    y_bl = df['BL']

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    sns.despine(left=True, bottom=True)
    fig.suptitle('')
    i = 0
    for a in ax.ravel():
        l1 = a.plot(y_val, label="labels")[0]
        l3 = a.plot(y_bl, label='base line')[0]
        l2 = a.plot(df[models[i]], label=models[i])

        a.set_ylim([0, 8])
        a.set_title(f'{models[i]}')
        i += 1

    fig.legend([l1, l3, l2],
               labels=['label', 'base line', 'model'],
               loc="lower center", ncol=3)

    for axis in fig.get_axes():
        axis.label_outer()

    plt.show()
    fig.savefig(f"{results_dir}/predictions.png")

    #scatter_plot
    x_uLSTM = df['uLSTM']
    plt.scatter(x_uLSTM, y_val)
    plt.xlabel('Y_hat')
    plt.ylabel('Y_Real')
    plt.legend()
    plt.plot(x_uLSTM, x_uLSTM, color='black', label= 'x=y')
    plt.show()

def plot_model_comparison(comparison_path, results_dir):
    results = pd.read_csv(comparison_path)
    # 1st figure - performance using different layers
    fig, axes = plt.subplots(1, 2)
    fig.suptitle("Model Performance Using Different Layers")
    color_dict_rmse = {"SNN_rmse":"#77CCFF", "uLSTM_rmse" : "#55AAFF", "BiLSTM_rmse" : "#3388FF", "Linear_rmse": "#0066FF", "BL_rmse" : "#0044FF"}
    results.plot.bar(x='Story', y=[col for col in results.columns if 'rmse' in col], color=color_dict_rmse, ax=axes[0])
    axes[0].set_title('RMSE')
    color_dict_r_square = {"SNN_r_square_correlation":"#77CCFF", "uLSTM_r_square_correlation" : "#55AAFF", "BiLSTM_r_square_correlation" : "#3388FF", "Linear_r_square_correlation": "#0066FF", "BL_r_square_correlation" : "#0044FF"}
    results.plot.bar(x='Story', y=[col for col in results.columns if 'r_square' in col], color= color_dict_r_square, ax=axes[1])
    axes[1].set_title('R Square')
    plt.rcParams.update({'font.size': 6})
    axes[0].tick_params(axis='x', labelsize=5.5)
    axes[1].tick_params(axis='x', labelsize=5.5)
    plt.tight_layout()
    fig.savefig(f"{results_dir}/layers.png")

    # 2nd figure - performance using different models
    fig, axes = plt.subplots(1, 2)
    fig.suptitle("Model Performance Using Different Models")

    rmse_df = results[[col for col in results.columns if 'rmse' in col]]
    rmse_df.columns = [col.split('_rmse')[0] for col in rmse_df.columns]
    rmse_df = rmse_df.join(results['Story'])
    rmse_t = rmse_df.transpose()
    rmse_t.columns = rmse_t.iloc[-1]
    rmse_t.drop(rmse_t.tail(1).index, inplace=True)
    story = rmse_df['Story']
    if len(story) == 1:
        rmse_t.plot.bar(ax=axes[0], color="#3388FF")
    else:
        color_dict = {story[0]: "#77CCFF", story[1]: "#55AAFF", story[2]: "#3388FF", story[3]: "#0066FF"}
        rmse_t.plot.bar(ax=axes[0], color=color_dict)
    axes[0].set_title('RMSE')

    rsq_df = results[[col for col in results.columns if 'r_square' in col]]
    rsq_df.columns = [col.split('_r_square')[0] for col in rsq_df.columns]
    rsq_df = rsq_df.join(results['Story'])
    rsq_t = rsq_df.transpose()
    rsq_t.columns = rsq_t.iloc[-1]
    rsq_t.drop(rsq_t.tail(1).index, inplace=True)
    if len(story) == 1:
        rsq_t.plot.bar(ax=axes[1], color="#3388FF")
    else:
        color_dict = {story[0]: "#77CCFF", story[1]: "#55AAFF", story[2]: "#3388FF", story[3]: "#0066FF"}
        rsq_t.plot.bar(ax=axes[1], color=color_dict)
    axes[1].set_title('R Square')
    plt.rcParams.update({'font.size': 6})
    fig.savefig(f"{results_dir}/models.png")


if __name__ == '__main__':
    plot_predictions('C:\\Users\\mayas\\PycharmProjects\\Sagol-project-emotion-classification\\test-plot.csv',
                     "C:\\Users\\mayas\\PycharmProjects\\Sagol-project-emotion-classification\\plots")
    plot_model_comparison('C:\\Users\\mayas\\PycharmProjects\\Sagol-project-emotion-classification\\comparison-test.csv'
                          , "C:\\Users\\mayas\\PycharmProjects\\Sagol-project-emotion-classification\\plots")
