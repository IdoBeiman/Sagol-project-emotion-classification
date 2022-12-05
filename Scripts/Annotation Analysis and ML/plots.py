import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_predictions(predictions_path, results_dir):
    sns.set()
    sns.set_theme(style="whitegrid", font="Times New Roman")
    sns.color_palette("bright")

    results = pd.read_csv(predictions_path)
    results.drop(['Unnamed: 0'], axis = 1, inplace=True)
    iter_num = len([val for val in results.columns.values if 'Real' in val])
    iter_cols = int(len(results.columns.values) / iter_num)
    df = results.iloc[:, 0: iter_cols]
    df.columns = [col.split('_')[0] for col in df.columns]
    for i in range(1, iter_num):
        new_df = results.iloc[:, 0+iter_cols*i:iter_cols+iter_cols*i]
        new_df.columns = [col.split('_')[0] for col in new_df.columns]
        df = pd.concat([df, new_df], ignore_index=True)

    models = [m for m in df.columns if m not in ["Baseline", "Real"]]

    y_val = df['Real']
    y_bl = df['Baseline']

    if len(models) % 2 == 0:
        if len(models) == 2:
            fig, ax = plt.subplots(1, 2, figsize=(15, 10), gridspec_kw={'hspace': 0.1, 'wspace': 0.1})
        else:
            num = len(models) // 2
            fig, ax = plt.subplots(2, num, figsize=(15, 10), gridspec_kw={'hspace': 0.1, 'wspace': 0.1})
    else:
        fig, ax = plt.subplots(1, len(models), figsize=(15, 10), gridspec_kw={'hspace': 0.1, 'wspace': 0.1})
    sns.despine(left=True, bottom=True)
    fig.suptitle('')
    i = 0
    # for a in ax.ravel():
    l1 = ax.plot(y_val, label="labels")[0]
    l3 = ax.plot(y_bl, label='base line')[0]
    l2 = ax.plot(df[models[i]], label=models[i])

    ax.set_ylim([0, 8])
    ax.set_title(f'{models[i]}')
    i += 1

    fig.legend([l1, l3, l2],
               labels=['label', 'base line', 'model'],
               loc="lower center", ncol=3)

    for axis in fig.get_axes():
        axis.label_outer()

    # plt.show()
    fig.savefig(f"{results_dir}/predictions.png")

    #scatter_plot
    #x_uLSTM = df['uLSTM']
    #plt.scatter(x_uLSTM, y_val)
    #plt.xlabel('Y_hat')
    #plt.ylabel('Y_Real')
    #plt.legend()
    #plt.plot(x_uLSTM, x_uLSTM, color='black', label= 'x=y')
    #plt.show()


def plot_model_comparison(comparison_path, results_dir):
    results = pd.read_csv(comparison_path)
    colors_dict = {"#77CCFF", "#55AAFF", "#3388FF", "#0066FF", "#0044FF"}
    fig1, axes = plt.subplots(1, 1)
    fig1.suptitle("Model Performance Using RMSE")
    results.plot.bar(x='story', y=[col for col in results.columns if 'rmse' in col], color=colors_dict, ax=axes)
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig1.savefig(f"{results_dir}/RMSE_performance.png")# 1st figure - performance using different layers
    fig, axes = plt.subplots(1, 2)
    fig.suptitle("Model Performance Using Different Layers")
    results.plot.bar(x='story', y=[col for col in results.columns if 'rmse' in col], color=colors_dict, ax=axes[0])
    axes[0].set_title('RMSE')
    results.plot.bar(x='story', y=[col for col in results.columns if 'r_square' in col], color= colors_dict, ax=axes[1])
    axes[1].set_title('R Square')
    plt.rcParams.update({'font.size': 6})
    axes[0].tick_params(axis='x', labelrotation=0)
    axes[1].tick_params(axis='x', labelrotation=0)
    plt.tight_layout()
    fig.savefig(f"{results_dir}/layers.png")

    # 2nd figure - performance using different models
    fig, axes = plt.subplots(1, 2)
    fig.suptitle("Model Performance Using Different Models", fontsize=12)

    rmse_df = results[[col for col in results.columns if 'rmse' in col]]
    rmse_df.columns = [col.split('_rmse')[0] for col in rmse_df.columns]
    rmse_df = rmse_df.join(results['story'])
    rmse_t = rmse_df.transpose()
    rmse_t.columns = rmse_t.iloc[-1]
    rmse_t.drop(rmse_t.tail(1).index, inplace=True)
    story = rmse_df['story']
    rmse_t.plot.bar(ax=axes[0], color=colors_dict)
    axes[0].set_title('RMSE')

    rsq_df = results[[col for col in results.columns if 'r_square' in col]]
    rsq_df.columns = [col.split('_r_square')[0] for col in rsq_df.columns]
    rsq_df = rsq_df.join(results['story'])
    rsq_t = rsq_df.transpose()
    rsq_t.columns = rsq_t.iloc[-1]
    rsq_t.drop(rsq_t.tail(1).index, inplace=True)
    rsq_t.plot.bar(ax=axes[1], color=colors_dict)
    axes[1].set_title('R Square')
    plt.rcParams.update({'font.size': 6})
    plt.tight_layout()
    fig.savefig(f"{results_dir}/models.png")
